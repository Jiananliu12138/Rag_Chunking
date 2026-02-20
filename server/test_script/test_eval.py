"""
Evaluation 相关接口测试脚本。

使用方式（先启动服务）：

    cd F:\thesis\Meta-Chunking\server
    uvicorn app.main:app --reload --port 8080

在项目根目录执行：

    python server/test_script/test_eval.py

环境变量：
    META_CHUNKING_BASE_URL  默认 http://localhost:8080/api/v1
"""
import json
import os
from typing import Any, Dict

import requests

BASE_URL = os.getenv("META_CHUNKING_BASE_URL", "http://localhost:8080/api/v1")

# 测试数据文件路径（按本地环境修改）
# 获取项目根目录（server/test_script/.. -> server/.. -> 项目根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SAMPLE_RESULTS_JSON = os.path.join(PROJECT_ROOT, "eval", "LongBench", "sample_results.json")
# 如果 sample_results.json 不存在，会在脚本中创建示例文件
TEST_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_script")


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
            # 如果请求体太长，只显示前 500 字符
            if len(body) > 500:
                print(body[:500] + "...")
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


def ensure_sample_results_json() -> str:
    """确保 sample_results.json 存在，不存在则创建示例文件。"""
    path = SAMPLE_RESULTS_JSON
    if os.path.isfile(path):
        return path

    # 创建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 创建示例数据
    sample_data = [
        {
            "_id": "q1",
            "input": "What is machine learning?",
            "llm_ans": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "answers": ["Machine learning is a method of data analysis that automates analytical model building."],
            "retrieval_list": [
                "Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
                "It focuses on the development of computer programs that can access data and use it to learn for themselves.",
            ],
        },
        {
            "_id": "q2",
            "input": "What is deep learning?",
            "llm_ans": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns.",
            "answers": ["Deep learning is part of machine learning methods based on artificial neural networks."],
            "retrieval_list": [
                "Deep learning is a subset of machine learning in artificial intelligence (AI) that has networks capable of learning unsupervised from data that is unstructured or unlabeled.",
                "Also known as deep neural learning or deep neural network.",
            ],
        },
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    print(f"已创建示例文件: {path}\n")
    return path


def test_traditional_eval() -> None:
    """测试 POST /eval/traditional — 传统指标评估（直接传数据）。"""
    url = f"{BASE_URL}/eval/traditional"
    payload: Dict[str, Any] = {
        "predictions": [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
        ],
        "answers": [
            ["Machine learning is a method of data analysis."],
            ["Deep learning is part of machine learning methods."],
        ],
        # BERTScore 参数可选，未提供时从配置读取
        "enable_bert_score": False,  # 设为 True 需要 GPU
    }
    resp = requests.post(url, json=payload, timeout=300)
    _print_response(resp)


def test_traditional_eval_with_test_field() -> None:
    """测试 POST /eval/traditional — 使用 test 字段（sample_results.json 格式）。"""
    url = f"{BASE_URL}/eval/traditional"
    payload: Dict[str, Any] = {
        "test": [
            {
                "_id": "q1",
                "input": "What is AI?",
                "llm_ans": "Artificial Intelligence is the simulation of human intelligence.",
                "answers": ["AI is the ability of machines to perform tasks that typically require human intelligence."],
            },
            {
                "_id": "q2",
                "input": "What is NLP?",
                "llm_ans": "Natural Language Processing is a branch of AI.",
                "answers": ["NLP is a field of AI that focuses on the interaction between computers and human language."],
            },
        ],
        "enable_bert_score": False,
    }
    resp = requests.post(url, json=payload, timeout=300)
    _print_response(resp)


def test_traditional_eval_file() -> None:
    """测试 POST /eval/traditional-file — 传统指标评估（从文件读取）。"""
    input_path = ensure_sample_results_json()
    url = f"{BASE_URL}/eval/traditional-file"
    payload: Dict[str, Any] = {
        "input_path": input_path,
        # BERTScore 参数可选，未提供时从配置读取
        "enable_bert_score": False,
    }
    resp = requests.post(url, json=payload, timeout=300)
    _print_response(resp)


def test_ragas_eval() -> None:
    """测试 POST /eval/ragas — RAGAS 评估（直接传数据）。

    注意：需要配置 vLLM 服务和嵌入模型路径（通过环境变量或请求参数）。
    """
    url = f"{BASE_URL}/eval/ragas"
    payload: Dict[str, Any] = {
        "dataset": {
            "question": [
                "What is machine learning?",
                "What is deep learning?",
            ],
            "answer": [
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with multiple layers.",
            ],
            "contexts": [
                [
                    "Machine learning is a subset of AI that enables systems to learn from experience.",
                    "It focuses on developing computer programs that can access data and learn for themselves.",
                ],
                [
                    "Deep learning is a subset of machine learning using neural networks.",
                    "It uses multiple layers to model and understand complex patterns.",
                ],
            ],
            "ground_truth": [
                "Machine learning is a method of data analysis that automates analytical model building.",
                "Deep learning is part of machine learning methods based on artificial neural networks.",
            ],
        },
        # 以下参数均为可选，未提供时从环境变量/配置读取
        # "vllm_api_base": "http://localhost:8005/v1",
        # "vllm_api_key": "EMPTY",
        # "vllm_model_name": "Qwen2.5-7B-Instruct",
        # "embedding_model_path": "/path/to/bge-large-en-v1.5",
        # "device": "cuda:0",
        # "enable_cache": True,
        # "cache_dir": "./ragas_cache",
    }
    resp = requests.post(url, json=payload, timeout=600)
    _print_response(resp)


def test_ragas_eval_file() -> None:
    """测试 POST /eval/ragas-file — RAGAS 评估（从文件读取）。

    支持两种文件格式：
    1. 标准 RAGAS 格式：{"question": [...], "answer": [...], "contexts": [...], "ground_truth": [...]}
    2. sample_results.json 格式：列表，每项包含 input/llm_ans/answers/retrieval_list

    注意：需要配置 vLLM 服务和嵌入模型路径（通过环境变量或请求参数）。
    """
    # 使用 sample_results.json 格式
    input_path = ensure_sample_results_json()

    url = f"{BASE_URL}/eval/ragas-file"
    payload: Dict[str, Any] = {
        "input_path": input_path,
        # 以下参数均为可选，未提供时从环境变量/配置读取
        # "vllm_api_base": "http://localhost:8005/v1",
        # "vllm_api_key": "EMPTY",
        # "vllm_model_name": "Qwen2.5-7B-Instruct",
        # "embedding_model_path": "/path/to/bge-large-en-v1.5",
        # "device": "cuda:0",
        # "enable_cache": True,
        # "cache_dir": "./ragas_cache",
    }
    resp = requests.post(url, json=payload, timeout=600)
    _print_response(resp)


def create_ragas_format_json() -> str:
    """创建标准 RAGAS 格式的 JSON 文件用于测试。"""
    path = os.path.join(TEST_OUTPUT_DIR, "ragas_dataset.json")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    ragas_data = {
        "question": [
            "What is machine learning?",
            "What is deep learning?",
        ],
        "answer": [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
        ],
        "contexts": [
            [
                "Machine learning is a subset of AI that enables systems to learn from experience.",
                "It focuses on developing computer programs that can access data and learn for themselves.",
            ],
            [
                "Deep learning is a subset of machine learning using neural networks.",
                "It uses multiple layers to model and understand complex patterns.",
            ],
        ],
        "ground_truth": [
            "Machine learning is a method of data analysis that automates analytical model building.",
            "Deep learning is part of machine learning methods based on artificial neural networks.",
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(ragas_data, f, ensure_ascii=False, indent=2)
    print(f"已创建 RAGAS 格式文件: {path}\n")
    return path


def test_ragas_eval_file_standard_format() -> None:
    """测试 POST /eval/ragas-file — 使用标准 RAGAS 格式文件。"""
    input_path = create_ragas_format_json()
    url = f"{BASE_URL}/eval/ragas-file"
    payload: Dict[str, Any] = {
        "input_path": input_path,
    }
    resp = requests.post(url, json=payload, timeout=600)
    _print_response(resp)


if __name__ == "__main__":
    print(f"BASE_URL = {BASE_URL}")
    print(f"SAMPLE_RESULTS_JSON = {SAMPLE_RESULTS_JSON}")
    print()

    # 1. 传统指标评估（直接传数据）
    print("=" * 80)
    print("测试 1: POST /eval/traditional (直接传 predictions/answers)")
    print("=" * 80)
    test_traditional_eval()

    # 2. 传统指标评估（使用 test 字段）
    print("=" * 80)
    print("测试 2: POST /eval/traditional (使用 test 字段)")
    print("=" * 80)
    test_traditional_eval_with_test_field()

    # 3. 传统指标评估（从文件读取）
    print("=" * 80)
    print("测试 3: POST /eval/traditional-file")
    print("=" * 80)
    test_traditional_eval_file()

    # 4. RAGAS 评估（直接传数据）
    print("=" * 80)
    print("测试 4: POST /eval/ragas (直接传 dataset)")
    print("=" * 80)
    print("注意：需要配置 vLLM 服务和嵌入模型，否则会报错")
    # test_ragas_eval()  # 需要 vLLM，默认注释

    # 5. RAGAS 评估（从文件读取，sample_results.json 格式）
    print("=" * 80)
    print("测试 5: POST /eval/ragas-file (sample_results.json 格式)")
    print("=" * 80)
    print("注意：需要配置 vLLM 服务和嵌入模型，否则会报错")
    # test_ragas_eval_file()  # 需要 vLLM，默认注释

    # 6. RAGAS 评估（从文件读取，标准 RAGAS 格式）
    print("=" * 80)
    print("测试 6: POST /eval/ragas-file (标准 RAGAS 格式)")
    print("=" * 80)
    print("注意：需要配置 vLLM 服务和嵌入模型，否则会报错")
    # test_ragas_eval_file_standard_format()  # 需要 vLLM，默认注释

    print("\n所有测试完成！")
    print("\n提示：")
    print("- 传统指标评估（F1/ROUGE/BLEU）不需要额外服务，可以直接运行")
    print("- RAGAS 评估需要 vLLM 服务和嵌入模型，请先配置环境变量后再取消注释相关测试")
