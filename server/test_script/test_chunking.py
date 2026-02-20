"""
简单的 chunking 接口测试脚本。

使用方式（先在根目录启动服务，例如 uvicorn app.main:app --reload --port 8080）：

    (rag) F:\thesis\Meta-Chunking>conda activate rag
    (rag) F:\thesis\Meta-Chunking>python test_script/test_chunking.py

如需切换服务地址，可以设置环境变量：

    META_CHUNKING_BASE_URL=http://localhost:8080/api/v1
"""

import json
import os
from typing import Any, Dict

import requests


BASE_URL = os.getenv("META_CHUNKING_BASE_URL", "http://localhost:8080/api/v1")

# 这里按你本地实际情况修改
CHUNK_INPUT_FILE = r"/data/h50056789/Rag_Chunking/Corpus/LongBench/2wikimqa.jsonl"
CHUNK_OUTPUT_DIR = r"/data/h50056789/Rag_Chunking/test_script"


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


def test_chunk_text_token() -> None:
    """测试 /chunks/chunk-text，token 分块。"""
    url = f"{BASE_URL}/chunks/chunk-text"
    payload: Dict[str, Any] = {
        "text": "这是一个用于测试 token 分块的示例文本。\n\n第二段内容在这里。",
        "method": "token",
        "token_params": {
            "chunk_token_size": 1200,
            "chunk_overlap_token_size": 100,
            "split_by_character": "\n\n",
        },
    }
    resp = requests.post(url, json=payload, timeout=300)
    _print_response(resp)


def test_chunk_text_llamaindex() -> None:
    """测试 /chunks/chunk-text，llamaindex 分块。"""
    url = f"{BASE_URL}/chunks/chunk-text"
    payload: Dict[str, Any] = {
        "text": "LlamaIndex SimpleNodeParser 测试文本。用于验证 llamaindex 分块是否正常工作。",
        "method": "llamaindex",
        "llamaindex_params": {
            "chunk_size": 512,
            "chunk_overlap": 50,
        },
    }
    resp = requests.post(url, json=payload, timeout=300)
    _print_response(resp)


def test_chunk_text_semantic() -> None:
    """测试 /chunks/chunk-text，semantic 分块。

    注意：嵌入模型路径从环境变量 DEFAULT_EMBEDDING_MODEL 读取，
    如未配置会在服务端报错。
    """
    url = f"{BASE_URL}/chunks/chunk-text"
    payload: Dict[str, Any] = {
        "text": (
            "气候变化是当今世界面临的最严峻挑战之一。\n\n"
            "可再生能源包括太阳能、风能等形式，它们能够在减少碳排放的同时满足人类用能需求。"
        ),
        "method": "semantic",
        # semantic_params 可留空，直接用环境默认模型
        "semantic_params": {
            "buffer_size": 1,
            "breakpoint_percentile_threshold": 74,
        },
    }
    resp = requests.post(url, json=payload, timeout=600)
    _print_response(resp)


def test_chunk_text_lumber() -> None:
    """测试 /chunks/chunk-text，lumber 分块。

    vLLM 相关配置从环境变量 DEFAULT_VLLM_API_BASE / DEFAULT_VLLM_MODEL_NAME
    等读取，如未配置会在服务端报错。
    """
    url = f"{BASE_URL}/chunks/chunk-text"
    payload: Dict[str, Any] = {
        "text": (
            "第一章 机器学习概述。\n\n"
            "第二章 深度神经网络。\n\n"
            "第三章 自然语言处理基础。"
        ),
        "method": "lumber",
        # lumber_params 可省略或只按需覆盖部分字段
        "lumber_params": {},
    }
    resp = requests.post(url, json=payload, timeout=1200)
    _print_response(resp)


def test_chunk_file_llamaindex() -> None:
    """测试 /chunks/chunk-file，llamaindex 对文件分块。

    CHUNK_INPUT_FILE: 需要是一个 jsonl 或兼容格式的输入文件，
    CHUNK_OUTPUT_DIR: 输出目录，会在其中生成分块结果文件。
    """
    url = f"{BASE_URL}/chunks/chunk-file"
    payload: Dict[str, Any] = {
        "method": "llamaindex",
        "input_file": CHUNK_INPUT_FILE,
        "output_dir": CHUNK_OUTPUT_DIR,
        "llamaindex_params": {
            "chunk_size": 512,
            "chunk_overlap": 50,
        },
    }
    resp = requests.post(url, json=payload, timeout=3600)
    _print_response(resp)


def test_list_methods() -> None:
    """测试 /chunks/methods，查看当前支持的所有分块方法及参数 schema。"""
    url = f"{BASE_URL}/chunks/methods"
    resp = requests.get(url, timeout=60)
    _print_response(resp)


if __name__ == "__main__":
    print(f"使用 BASE_URL = {BASE_URL}")
    print("请先根据实际情况修改 CHUNK_INPUT_FILE / CHUNK_OUTPUT_DIR 再运行。\n")

    # 1. 查看支持的方法
    test_list_methods()

    # 2. 文本分块示例（可以按需注释掉某些调用）
    # test_chunk_text_token()
    # test_chunk_text_llamaindex()
    # 如环境已配置好语义模型 / vLLM，再开启下面两行：
    # test_chunk_text_semantic()
    # test_chunk_text_lumber()

    # 3. 文件分块示例（记得改成你自己的输入/输出路径）
    test_chunk_file_llamaindex()

