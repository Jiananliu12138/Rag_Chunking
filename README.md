# Meta-Chunking

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688)
![React](https://img.shields.io/badge/React%20%2B%20Vite-frontend-61DAFB)
![Milvus](https://img.shields.io/badge/Milvus-vector%20database-00A1EA)
![License](https://img.shields.io/badge/License-Apache--2.0-green)

Meta-Chunking is a research-oriented RAG platform for experimenting with document chunking, vector indexing, retrieval, generation, and evaluation. The project packages several chunking strategies into a usable workflow, with a FastAPI backend and a React frontend for running experiments without repeatedly writing glue code.

The main goal is to make chunking methods easier to compare in a full RAG pipeline: split documents, build a vector index, retrieve or generate answers, then evaluate both the final answers and the chunking component itself.

## Features

- Multiple chunking strategies: token-based chunking, semantic chunking, LlamaIndex fixed-window chunking, and LLM-driven Lumber-style chunking.
- Vector indexing with Milvus / Milvus Lite, including collection management and metadata-aware deletion.
- Retrieval and RAG generation with OpenAI-compatible LLM endpoints such as vLLM.
- Hybrid search support with dense + sparse retrieval and RRF-style rank fusion.
- Optional reranking through a local CrossEncoder or a vLLM-compatible rerank endpoint.
- End-to-end evaluation with traditional QA metrics, LLM Judge, RAGAS, and retrieval metrics.
- Component-level chunk evaluation, including chunk quality and chunk stickiness.
- RAGAS-based testset generation from existing chunk files.
- Web UI for chunking, indexing, retrieval, evaluation, and testset generation.

## Architecture

```text
Raw corpus
   |
   v
Chunking methods
   |-- token
   |-- semantic
   |-- llamaindex
   |-- lumber
   |
   v
Chunk result files
   |
   v
Milvus index  --->  Retrieval / Hybrid Search / Rerank
   |                                  |
   |                                  v
   |                            RAG Generation
   |                                  |
   v                                  v
Component Evaluation          End-to-End Evaluation
```

## Project Structure

```text
.
+-- server/                    # FastAPI backend
|   +-- app/api/v1/             # REST API controllers
|   +-- app/services/           # Chunking, indexing, retrieval, eval services
|   +-- app/repositories/       # Milvus and file access layer
|   +-- app/schemas/            # Pydantic request/response schemas
|   +-- run.py                  # Local backend launcher
+-- frontend/                   # React + Vite web interface
|   +-- src/app/pages/          # Main pages: chunking, index, retrieval, eval
+-- public_method/              # Core reusable algorithms
|   +-- chunking/               # Token, semantic, LlamaIndex, Lumber chunkers
|   +-- evaluation/             # LongBench/RAGAS/component/end-to-end metrics
+-- generate_testset/           # Standalone RAGAS testset generation scripts
+-- QA_corpus/                  # QA extraction utilities
+-- eval/                       # Evaluation workspace / outputs
+-- docker-compose.yml          # Optional Milvus standalone stack
+-- requirements.txt            # Python dependencies
+-- start_vllm.sh               # Example vLLM startup script
```

## Quick Start

### 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

GPU-related packages such as `torch`, `vllm`, `sentence-transformers`, and model downloads may need to be adjusted for your CUDA environment.

### 2. Configure runtime variables

Create or edit `.env` in the project root. The most important options are:

```env
MILVUS_DATA_DIR=./milvus_data
MILVUS_URI=
TIKTOKEN_CACHE_DIR=./tiktoken_cache

DEFAULT_EMBEDDING_MODEL=/path/to/BAAI/bge-m3
DEFAULT_EMBEDDING_DIM=1024
DEFAULT_EMBEDDING_DEVICE=cuda:0
DEFAULT_EMBEDDING_BASE=http://localhost:8003/v1
DEFAULT_EMBEDDING_NAME=BAAI/bge-m3

DEFAULT_LLM_API_BASE=http://localhost:8001/v1
DEFAULT_LLM_API_KEY=EMPTY
DEFAULT_LLM_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8

DEFAULT_RERANK_MODEL=/path/to/BAAI/bge-reranker-v2-m3
DEFAULT_RERANK_DEVICE=cuda:0
```

By default, the backend can use local Milvus Lite files when `MILVUS_URI` is empty. If you want to use a standalone Milvus service, start it with Docker and set `MILVUS_URI`.

```bash
docker compose up -d
```

### 3. Start an OpenAI-compatible LLM service

The backend expects OpenAI-compatible endpoints for generation, semantic chunking embeddings, and optional reranking. vLLM is one common choice:

```bash
bash start_vllm.sh
```

Edit `start_vllm.sh` to match your model path, GPU, port, and context length.

### 4. Run the backend

```bash
cd server
python run.py
```

The API will be available at:

- Backend: `http://127.0.0.1:8081`
- Swagger UI: `http://127.0.0.1:8081/docs`
- API prefix: `http://127.0.0.1:8081/api/v1`

### 5. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend uses `http://127.0.0.1:8081/api/v1` by default. To override it:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8081/api/v1 npm run dev
```

## Main Workflow

1. Prepare a corpus file, usually JSONL.
2. Run a chunking method from the web UI or `/api/v1/chunks/*`.
3. Build a Milvus collection from the generated chunk result file.
4. Run retrieval or RAG generation against the collection.
5. Evaluate the output with retrieval metrics, traditional QA metrics, RAGAS, or LLM Judge.
6. Use component evaluation to analyze the chunking quality independently of the downstream generation model.

## Core Modules

### Chunking

Located in `public_method/chunking/` and exposed by `server/app/services/chunk_service.py`.

Supported methods:

- `token`: fixed token-size chunking based on `tiktoken`, with overlap and delimiter options.
- `semantic`: semantic breakpoint chunking using an embedding model.
- `llamaindex`: LlamaIndex `SimpleNodeParser` style fixed-window chunking.
- `lumber`: LLM-assisted chunking that detects topic or discourse boundaries through a vLLM/OpenAI-compatible model.

### Indexing

Located in `server/app/services/index_service.py` and `server/app/repositories/milvus_repository.py`.

Indexing reads chunk result files, embeds text chunks, and writes them into Milvus collections. The service also supports:

- building or adding to a collection,
- listing and inspecting collections,
- deleting collections,
- deleting vectors by metadata such as `filepath` or `doc_id`.

### Retrieval and Generation

Located in `server/app/services/retrieval_service.py`.

The retrieval module supports vector search, hybrid dense/sparse search, optional reranking, and full RAG generation. Batch generation from JSONL files is also supported for experiment pipelines.

### Evaluation

Located in `public_method/evaluation/` and `server/app/services/eval_service.py`.

Supported evaluation paths include:

- traditional metrics: F1, ROUGE-L, BLEU-1/2/3/4, optional BERTScore;
- LLM Judge for answer equivalence scoring;
- RAGAS metrics such as faithfulness, answer relevancy, context recall, and context precision;
- retrieval metrics for ranked context evaluation.

### Component Evaluation

Located in `public_method/evaluation/component/`.

This module evaluates the chunking component directly, without relying only on final answer quality. It includes boundary clarity, semantic similarity, perplexity-based scoring, and graph-based chunk stickiness analysis.

### Testset Generation

Located in `server/app/services/testset/` and `generate_testset/`.

This module generates synthetic QA testsets from chunk files using RAGAS and an OpenAI-compatible LLM. It can be called through the API, the frontend, or standalone scripts such as:

```bash
python generate_testset/run_generate_with_chunks.py --input /path/to/chunks.json --output /path/to/testset.jsonl
```

## API Overview

The backend is organized under `/api/v1`:

| Module | Endpoint Prefix | Description |
| --- | --- | --- |
| Chunking | `/chunks` | Chunk text or files with different strategies |
| Index | `/index` | Build, add, inspect, and delete Milvus collections |
| Retrieval | `/retrieval` | Search, RAG generation, and batch generation |
| Testset | `/testset` | Generate QA testsets from chunks |
| Evaluation | `/eval` | Traditional, RAGAS, and retrieval evaluation |
| Component Eval | `/component-eval` | Chunk quality and stickiness evaluation |
| Files | `/files` | Browse local files from the frontend |

For request schemas and examples, open `http://127.0.0.1:8081/docs` after starting the backend.

## Notes

- The project is designed for research and experimentation. Some defaults in `.env`, frontend runtime defaults, and shell scripts use local absolute paths; change them before running on a new machine.
- For reproducible experiments, keep the chunking parameters, embedding model, embedding dimension, and collection name together in your experiment logs.
- When using reranking or RAGAS, model latency can dominate runtime. Tune worker counts in `.env` based on available GPU and API throughput.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
