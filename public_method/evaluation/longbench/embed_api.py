import json
import logging
from dataclasses import dataclass
from typing import Optional

import requests


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    # Backend FastAPI base url (no trailing slash)
    API_BASE: str = "http://127.0.0.1:8000"

    # "build" -> /api/v1/index/build, "add" -> /api/v1/index/add
    MODE: str = "build"

    COLLECTION_NAME: str = "lumber_chunk"
    # Example input:
    # F:\thesis\Meta-Chunking\MoC\our_metrics\test_data\test_add.json
    DOCS_PATH: str = r"F:\thesis\Meta-Chunking\MoC\our_metrics\test_data\test_add.json"

    EMBED_MODEL_PATH: str = r"/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5"
    EMBED_DIM: int = 1024
    BATCH_SIZE: int = 100

    # Build-only switch: None means backend default; True/False means explicit.
    ENABLE_SPARSE: Optional[bool] = True

    TIMEOUT_SECONDS: int = 1800


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    mode = Config.MODE.strip().lower()
    if mode not in {"build", "add"}:
        raise ValueError(f"Unsupported MODE={Config.MODE}. Expected 'build' or 'add'.")

    endpoint = "/api/v1/index/build" if mode == "build" else "/api/v1/index/add"
    url = f"{Config.API_BASE}{endpoint}"

    payload = {
        "collection_name": Config.COLLECTION_NAME,
        "docs_path": Config.DOCS_PATH,
        "batch_size": Config.BATCH_SIZE,
        "embed_model_path": Config.EMBED_MODEL_PATH,
        "embed_dim": Config.EMBED_DIM,
    }
    if mode == "build":
        payload["enable_sparse"] = Config.ENABLE_SPARSE

    logger.info("Calling %s", url)
    logger.info("collection=%s docs_path=%s", Config.COLLECTION_NAME, Config.DOCS_PATH)

    result = _post_json(url, payload, timeout=Config.TIMEOUT_SECONDS)
    if not result.get("success", False):
        raise RuntimeError(f"API failed: {json.dumps(result, ensure_ascii=False)}")

    data = result.get("data") or {}
    logger.info("Done: %s", result.get("message", "ok"))
    logger.info(
        "collection=%s uri=%s docs=%d filepaths=%d",
        data.get("collection_name"),
        data.get("milvus_uri"),
        len(data.get("doc_ids", [])),
        len(data.get("filepaths", [])),
    )
    logger.info("doc_chunk_map keys=%d", len((data.get("doc_chunk_map") or {}).keys()))


if __name__ == "__main__":
    main()
