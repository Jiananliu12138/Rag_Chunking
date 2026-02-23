"""
检索与 RAG 条件查询测试脚本。

覆盖接口：
- POST /retrieval/search
- POST /retrieval/generate

重点测试：
- 按 filepath 过滤（只检索某个源文件的 chunks）
- 按 doc_id 过滤（只检索某个 doc_id 对应的 chunks）

使用方式（先启动服务）：

    cd F:\thesis\Meta-Chunking\server
    uvicorn app.main:app --reload --port 8080

在项目根目录执行：

    python server/test_script/test_retrieval_filter.py

环境变量：
    META_CHUNKING_BASE_URL  默认 http://localhost:8080/api/v1
"""

import json
import os
from typing import Any, Dict

import requests

BASE_URL = os.getenv("META_CHUNKING_BASE_URL", "http://localhost:8080/api/v1")

# 获取项目根目录（server/test_script/.. -> server/.. -> 项目根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# 这里按你本地索引构建时的实际情况修改：
DEFAULT_COLLECTION_NAME = "test_chunks"
DEFAULT_EMBED_MODEL_PATH = os.getenv(
    "DEFAULT_EMBEDDING_MODEL",
    "/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5",
)
DEFAULT_EMBED_DIM = int(os.getenv("DEFAULT_EMBEDDING_DIM", "1024"))

# 用于演示条件过滤的示例值：
# - filepath：对应 MoC/our_metrics/test_data/test.json 里的 filepath 字段
# - doc_id：对应 splits 第二列（比如 1）
SAMPLE_FILEPATH = "./dataset/docs/2wikimqa/0a64d8873482d91efc595a508218c6ce881c13c95028039e.txt"
SAMPLE_DOC_ID = 1


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


def test_search_no_filter() -> None:
    """测试 /retrieval/search —— 不带任何过滤条件，全库检索。"""
    url = f"{BASE_URL}/retrieval/search"
    payload: Dict[str, Any] = {
        "query": "Who is Peter Rosegger?",
        "collection_name": DEFAULT_COLLECTION_NAME,
        "embed_model_path": DEFAULT_EMBED_MODEL_PATH,
        "embed_dim": DEFAULT_EMBED_DIM,
        "top_k": 5,
    }
    resp = requests.post(url, json=payload, timeout=300)
    _print_response(resp)


def test_search_with_filepath() -> None:
    """测试 /retrieval/search —— 按 filepath 过滤，只检索指定文件。"""
    url = f"{BASE_URL}/retrieval/search"
    payload: Dict[str, Any] = {
        "query": "Who is Peter Rosegger?",
        "collection_name": DEFAULT_COLLECTION_NAME,
        "embed_model_path": DEFAULT_EMBED_MODEL_PATH,
        "embed_dim": DEFAULT_EMBED_DIM,
        "top_k": 5,
        "filepath": SAMPLE_FILEPATH,
    }
    resp = requests.post(url, json=payload, timeout=300)
    _print_response(resp)


def test_search_with_doc_id() -> None:
    """测试 /retrieval/search —— 按 doc_id 过滤，只检索指定文档 ID。"""
    url = f"{BASE_URL}/retrieval/search"
    payload: Dict[str, Any] = {
        "query": "Who is Peter Rosegger?",
        "collection_name": DEFAULT_COLLECTION_NAME,
        "embed_model_path": DEFAULT_EMBED_MODEL_PATH,
        "embed_dim": DEFAULT_EMBED_DIM,
        "top_k": 5,
        "doc_id": SAMPLE_DOC_ID,
    }
    resp = requests.post(url, json=payload, timeout=300)
    _print_response(resp)


def test_rag_generate_with_filters() -> None:
    """测试 /retrieval/generate —— 带 filepath/doc_id 过滤的 RAG。"""
    url = f"{BASE_URL}/retrieval/generate"
    payload: Dict[str, Any] = {
        "query": "Who is Peter Rosegger?",
        "collection_name": DEFAULT_COLLECTION_NAME,
        "embed_model_path": DEFAULT_EMBED_MODEL_PATH,
        "embed_dim": DEFAULT_EMBED_DIM,
        "top_k": 3,
        # 下面两个条件参数可按需切换
        "filepath": SAMPLE_FILEPATH,
        "doc_id": SAMPLE_DOC_ID,
        # LLM 相关参数，按你本地 vLLM 环境调整
        "llm_api_base": os.getenv("DEFAULT_VLLM_API_BASE", "http://localhost:8005/v1"),
        "llm_model_name": os.getenv("DEFAULT_VLLM_MODEL_NAME", "Qwen2.5-7B-Instruct"),
        "temperature": 0.1,
        "max_new_tokens": 512,
    }
    resp = requests.post(url, json=payload, timeout=600)
    _print_response(resp)


if __name__ == "__main__":
    print(f"BASE_URL = {BASE_URL}")
    print(f"DEFAULT_COLLECTION_NAME = {DEFAULT_COLLECTION_NAME}")
    print(f"DEFAULT_EMBED_MODEL_PATH = {DEFAULT_EMBED_MODEL_PATH}")
    print(f"DEFAULT_EMBED_DIM = {DEFAULT_EMBED_DIM}")
    print(f"SAMPLE_FILEPATH = {SAMPLE_FILEPATH}")
    print(f"SAMPLE_DOC_ID = {SAMPLE_DOC_ID}")
    print()
    print("注意：请先用 /index/build 在 collection 中构建好索引，且 docs_path 对应的 JSON 中包含 filepath / doc_id 元数据。")
    print()

    print("=" * 80)
    print("测试 1: /retrieval/search —— 无过滤条件（全库检索）")
    print("=" * 80)
    test_search_no_filter()

    print("=" * 80)
    print("测试 2: /retrieval/search —— 按 filepath 过滤")
    print("=" * 80)
    test_search_with_filepath()

    print("=" * 80)
    print("测试 3: /retrieval/search —— 按 doc_id 过滤")
    print("=" * 80)
    test_search_with_doc_id()

    print("=" * 80)
    print("测试 4: /retrieval/generate —— 带过滤条件的 RAG 生成（需 vLLM）")
    print("=" * 80)
    # 如果本地未启动 vLLM，可暂时注释掉
    # test_rag_generate_with_filters()

    print("\n测试脚本结束。")

