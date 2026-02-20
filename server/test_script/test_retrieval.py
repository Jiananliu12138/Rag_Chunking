"""
Retrieval 相关接口测试脚本。

使用方式（先启动服务，并确保已有索引）：

    cd F:\thesis\Meta-Chunking\server
    uvicorn app.main:app --reload --port 8080

在项目根目录执行：

    python test_script/test_retrieval.py

环境变量：
    META_CHUNKING_BASE_URL  默认 http://localhost:8080/api/v1
"""
import json
import os
from typing import Any, Dict
import requests

BASE_URL = os.getenv("META_CHUNKING_BASE_URL", "http://localhost:8080/api/v1")

# 按本地环境修改：collection 需已通过 /index/build 构建
COLLECTION_NAME = "test_chunks"
EMBED_MODEL_PATH = os.getenv("DEFAULT_EMBEDDING_MODEL", "/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5")
EMBED_DIM = int(os.getenv("DEFAULT_EMBEDDING_DIM", "1024"))
# vLLM 服务地址与模型（/generate、/generate-file 需要）
LLM_API_BASE = os.getenv("DEFAULT_VLLM_API_BASE", "http://localhost:8005/v1")
LLM_MODEL_NAME = os.getenv("DEFAULT_VLLM_MODEL_NAME", "Qwen2.5-7B-Instruct")

# 文件批处理测试路径
RETRIEVAL_INPUT_JSONL = r"/data/h50056789/Rag_Chunking/eval/LongBench/sample_data.jsonl"
RETRIEVAL_OUTPUT_JSON = r"/data/h50056789/Rag_Chunking/test_script/retrieval_result.json"


def _print_response(resp: requests.Response) -> None:
    print("=" * 80)
    print(f"URL      : {resp.request.method} {resp.url}")
    print(f"Status   : {resp.status_code}")
    if resp.request.body:
        try:
            body = resp.request.body.decode("utf-8") if isinstance(resp.request.body, bytes) else resp.request.body
            print("Request  :", body[:500] + "..." if len(body) > 500 else body)
        except Exception:
            print("Request  : <body>")
    print("Response :")
    try:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        print(resp.text)
    print("=" * 80)
    print()


def test_search() -> None:
    """POST /retrieval/search — 向量相似度检索。"""
    url = f"{BASE_URL}/retrieval/search"
    payload: Dict[str, Any] = {
        "query": "What is the profession of Miloš Zličić?",
        "collection_name": COLLECTION_NAME,
        "embed_model_path": EMBED_MODEL_PATH,
        "embed_dim": EMBED_DIM,
        "top_k": 5,
    }
    resp = requests.post(url, json=payload, timeout=120)
    _print_response(resp)


def test_generate() -> None:
    """POST /retrieval/generate — 单条 RAG 检索+生成（需 vLLM 服务）。"""
    url = f"{BASE_URL}/retrieval/generate"
    payload: Dict[str, Any] = {
        "query": "What is the profession of Miloš Zličić?",
        "collection_name": COLLECTION_NAME,
        "embed_model_path": EMBED_MODEL_PATH,
        "embed_dim": EMBED_DIM,
        "top_k": 5,
        "llm_api_base": LLM_API_BASE,
        "llm_model_name": LLM_MODEL_NAME,
        "temperature": 0.1,
        "max_new_tokens": 256,
    }
    resp = requests.post(url, json=payload, timeout=180)
    _print_response(resp)


def test_generate_file() -> None:
    """POST /retrieval/generate-file — 从 jsonl 批处理 RAG，结果写 JSON。"""
    url = f"{BASE_URL}/retrieval/generate-file"
    payload: Dict[str, Any] = {
        "input_path": RETRIEVAL_INPUT_JSONL,
        "output_path": RETRIEVAL_OUTPUT_JSON,
        "collection_name": COLLECTION_NAME,
        "embed_model_path": EMBED_MODEL_PATH,
        "embed_dim": EMBED_DIM,
        "top_k": 5,
        "llm_api_base": LLM_API_BASE,
        "llm_model_name": LLM_MODEL_NAME,
        "temperature": 0.1,
        "max_new_tokens": 256,
    }
    resp = requests.post(url, json=payload, timeout=3600)
    _print_response(resp)


def ensure_sample_jsonl() -> None:
    """若 RETRIEVAL_INPUT_JSONL 不存在，写一个最小 sample 便于测试 generate-file。"""
    path = RETRIEVAL_INPUT_JSONL
    if os.path.isfile(path):
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    sample = [
        {"_id": "sample_1", "input": "What is a transformer in machine learning?", "answers": ["A transformer is..."]},
        {"_id": "sample_2", "input": "What is attention mechanism?", "answers": ["Attention allows..."]},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已生成示例 jsonl: {path}\n")


if __name__ == "__main__":
    print(f"BASE_URL = {BASE_URL}")
    print(f"COLLECTION_NAME = {COLLECTION_NAME}")
    print(f"EMBED_MODEL_PATH = {EMBED_MODEL_PATH}")
    print()

    # 1. 检索
    # test_search()

    # 2. 单条 RAG 生成（需 vLLM，未启动可注释）
    # test_generate()

    # 3. 文件批处理 RAG（需 vLLM + 输入 jsonl）
    ensure_sample_jsonl()
    test_generate_file()
