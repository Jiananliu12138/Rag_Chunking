"""
简单的向量索引接口测试脚本。

推荐流程：
1. 先用 chunking 接口对原始数据做分块，得到一个 JSON / JSONL 结果文件；
2. 确保该文件的格式能被 eval/LongBench/base_lite.py::_parse_chunks_from_json 正确解析；
3. 然后使用本脚本调用 /index/build 构建索引，再用 /index/add 做增量追加。

使用方式：

    (rag) F:\thesis\Meta-Chunking>conda activate rag
    (rag) F:\thesis\Meta-Chunking>python test_script/test_index.py

如需切换服务地址，可以设置环境变量：

    META_CHUNKING_BASE_URL=http://localhost:8080/api/v1
"""

import json
import os
from typing import Any, Dict

import requests


BASE_URL = os.getenv("META_CHUNKING_BASE_URL", "http://localhost:8080/api/v1")

# 下面这些参数请根据你本地实际情况修改
COLLECTION_NAME = "test_chunks"

# 用于 /index/build 的分块结果文件（通常是第一次构建）
DOCS_PATH_BUILD = r"F:\thesis\Meta-Chunking\your_chunk_result_for_build.json"

# 用于 /index/add 的分块结果文件（增量追加）
DOCS_PATH_ADD = r"F:\thesis\Meta-Chunking\your_chunk_result_for_add.json"


def _print_response(resp: requests.Response) -> None:
    """辅助打印 HTTP 响应。"""
    print("=" * 80)
    print(f"URL      : {resp.request.method} {resp.url}")
    print(f"Status   : {resp.status_code}")
    print("Request  :")
    try:
        if resp.request.body:
            body = (
                resp.request.body
                if isinstance(resp.request.body, (str, bytes))
                else json.dumps(resp.request.body)
            )
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


def test_build_index() -> None:
    """测试 /index/build，从分块结果文件构建/重建索引。"""
    url = f"{BASE_URL}/index/build"
    payload: Dict[str, Any] = {
        "collection_name": COLLECTION_NAME,
        "docs_path": DOCS_PATH_BUILD,
        "batch_size": 100,
    }
    resp = requests.post(url, json=payload, timeout=3600)
    _print_response(resp)


def test_add_index() -> None:
    """测试 /index/add，向已有索引追加数据。"""
    url = f"{BASE_URL}/index/add"
    payload: Dict[str, Any] = {
        "collection_name": COLLECTION_NAME,
        "docs_path": DOCS_PATH_ADD,
        "batch_size": 8000,
    }
    resp = requests.post(url, json=payload, timeout=3600)
    _print_response(resp)


def test_list_collections() -> None:
    """测试 /index/collections，列出所有 collection。"""
    url = f"{BASE_URL}/index/collections"
    resp = requests.get(url, timeout=60)
    _print_response(resp)


def test_delete_collection() -> None:
    """测试删除指定 collection（**危险操作，慎用**）。

    删除后对应的 Lite .db 文件会被移除，索引不可恢复。
    默认注释掉，如需测试请手动打开调用。
    """
    url = f"{BASE_URL}/index/collections/{COLLECTION_NAME}"
    resp = requests.delete(url, timeout=60)
    _print_response(resp)


if __name__ == "__main__":
    print(f"使用 BASE_URL = {BASE_URL}")
    print(
        "请先根据实际情况修改 COLLECTION_NAME / DOCS_PATH_BUILD / DOCS_PATH_ADD\n"
        "确保 docs_path 指向的文件格式与 eval/LongBench/base_lite.py::_parse_chunks_from_json 兼容。\n"
    )

    # 1. 构建索引（若已存在会覆盖）
    # 建议先确保对应 collection 不重要，或在测试环境中使用。
    test_build_index()

    # 2. 增量追加示例（如暂时没有追加文件，可以先注释掉）
    # test_add_index()

    # 3. 查看当前有哪些 collection
    test_list_collections()

    # 4. 如需测试删除，请手动取消下行注释（注意：操作不可逆）
    # test_delete_collection()

