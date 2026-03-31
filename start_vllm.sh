#!/bin/bash
# 强制使用第二张显卡 (GPU 1)
export CUDA_VISIBLE_DEVICES=1

# 启动 vLLM
python -m vllm.entrypoints.openai.api_server \
    --model /data/h50056789/Rag_Chunking/model/Qwen/Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5-7B-Instruct \
    --trust-remote-code \
    --port 8005 \
    --gpu-memory-utilization 0.83 \
    --max-model-len 65536 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --enforce-eager \
    --dtype bfloat16