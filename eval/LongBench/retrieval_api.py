import logging
from dataclasses import dataclass
from typing import Any

import requests


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    API_BASE: str = "http://127.0.0.1:8081"

    # Input can be jsonl (one query per line) or json list.
    # Each item supports:
    # - user_input / input / query
    # - _id / id
    # - answers (list[str]) or reference (str)
    INPUT_PATH: str = r"F:\thesis\Meta-Chunking\eval\LongBench\sample_data.jsonl"

    # generate-file API will write this file directly (final output)
    OUTPUT_PATH: str = r"F:\thesis\Meta-Chunking\eval\LongBench\sample_results_api.json"

    COLLECTION_NAME: str = "lumber_chunk"
    EMBED_MODEL_PATH: str = r"/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5"
    EMBED_DIM: int = 1024
    TOP_K: int = 5
    USE_HYBRID_SEARCH: bool | None = True
    # NOTE:
    # - 是否使用 Hybrid（dense+sparse）可在此脚本按请求控制（USE_HYBRID_SEARCH）。
    # - RRF 的具体类型与参数（ranker/k）由服务端配置控制：
    #   MILVUS_HYBRID_RANKER / MILVUS_HYBRID_RANKER_K。
    RERANK_ENABLED: bool = False
    RERANK_TYPE: str = "cross_encoder"
    RERANK_MODEL_PATH: str = ""
    RERANK_DEVICE: str = "cpu"
    RERANK_CANDIDATE_K: int | None = None
    RERANK_TOP_K: int | None = None

    LLM_API_BASE: str = "http://localhost:8005/v1"
    LLM_MODEL_NAME: str = r"/data/h50056789/Rag_Chunking/model/Qwen/Qwen2.5-7B-Instruct"
    TEMPERATURE: float = 0.1
    MAX_NEW_TOKENS: int = 1280

    TIMEOUT_SECONDS: int = 600


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    # generate-file API already writes the final JSON output file.
    endpoint = f"{Config.API_BASE}/api/v1/retrieval/generate-file"
    api_payload = {
        "input_path": Config.INPUT_PATH,
        "output_path": Config.OUTPUT_PATH,
        "collection_name": Config.COLLECTION_NAME,
        "embed_model_path": Config.EMBED_MODEL_PATH,
        "embed_dim": Config.EMBED_DIM,
        "top_k": Config.TOP_K,
        "use_hybrid_search": Config.USE_HYBRID_SEARCH,
        "rerank_enabled": Config.RERANK_ENABLED,
        "rerank_type": Config.RERANK_TYPE,
        "rerank_model_path": Config.RERANK_MODEL_PATH or None,
        "rerank_device": Config.RERANK_DEVICE,
        "rerank_candidate_k": Config.RERANK_CANDIDATE_K,
        "rerank_top_k": Config.RERANK_TOP_K,
        "llm_api_base": Config.LLM_API_BASE,
        "llm_model_name": Config.LLM_MODEL_NAME,
        "temperature": Config.TEMPERATURE,
        "max_new_tokens": Config.MAX_NEW_TOKENS,
    }

    logger.info("Calling %s", endpoint)
    result = _post_json(endpoint, api_payload, timeout=Config.TIMEOUT_SECONDS)
    if not result.get("success", False):
        raise RuntimeError(result.get("message", "unknown API error"))

    data: dict[str, Any] = result.get("data") or {}
    logger.info("Done: %s", result.get("message", "ok"))
    logger.info("Output file: %s", data.get("output_file"))
    logger.info("Processed=%s failed=%s", data.get("total_processed"), data.get("total_failed"))


if __name__ == "__main__":
    main()
