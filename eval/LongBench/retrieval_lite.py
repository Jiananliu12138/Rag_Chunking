"""
使用 Milvus Lite 的检索脚本（离线版本）
"""

import argparse
from loguru import logger
from base_lite import BaseRetrieverLite  # 改为使用 Milvus Lite
from embeddings.base import HuggingfaceEmbeddings
import json
import pandas as pd 
from transformers import AutoModelForCausalLM, AutoTokenizer
from llms.base import BaseLLM
import torch

class Qwen_7B_Chat(BaseLLM):
    def __init__(self, model_name='qwen_7b', temperature=1.0, max_new_tokens=1024):
        super().__init__(model_name, temperature, max_new_tokens)
        local_path = 'Qwen2-7B-Instruct'
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(local_path, device_map="auto",
                                                     trust_remote_code=True).eval()
        self.gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }

    def request(self, query: str) -> str:
        query = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        input_ids = self.tokenizer.encode(query, return_tensors="pt").cuda()
        output = self.model.generate(input_ids, **self.gen_kwargs)[0]
        response = self.tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response

class Baichuan2_7B_Chat(BaseLLM):
    def __init__(self, model_name='baichuan2_7b', temperature=1.0, max_new_tokens=1024):
        super().__init__(model_name, temperature, max_new_tokens)
        local_path = 'baichuan2-7b-chat'
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=True)
        self.gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }
    
    def request(self, query: str) -> str:
        input_ids = self.tokenizer.encode(query, return_tensors="pt").cuda()
        output = self.model.generate(input_ids, **self.gen_kwargs)[0]
        response = self.tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response

class GLM4_9B_Chat(BaseLLM):
    def __init__(self, model_name='glm4_9b', temperature=1.0, max_new_tokens=1024):
        super().__init__(model_name, temperature, max_new_tokens)
        local_path = 'glm-4-9b-chat'
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map="auto",
            torch_dtype=torch.float32,
            trust_remote_code=True)
        self.gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }
    
    def request(self, query: str) -> str:
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                            add_generation_prompt=True,
                                            tokenize=True,
                                            return_tensors="pt",
                                            return_dict=True
                                            )
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(**inputs, **self.gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response=self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

parser = argparse.ArgumentParser()
# 数据集参数
parser.add_argument('--data_path', default='data/qasper.jsonl', help="Path to the dataset")  
parser.add_argument('--save_file', default='qa_results/qasper_qwen2_7B_Chunks_310_2.json', help="Path to the answer")

# 嵌入模型参数
parser.add_argument('--embedding_name', default='BAAI/bge-large-en-v1.5')  
parser.add_argument('--embedding_dim', type=int, default=1024)   # bge-large-en-v1: 1024, bge-base-zh-v1.5: 768

# 文档和索引参数
parser.add_argument('--docs_path', default='chunk_otherllm_00/qasper_qwen2_7B_Chunks_310_2.json', help="Path to the retrieval documents")  
parser.add_argument('--construct_index', action='store_true', help="Whether to construct an index")
parser.add_argument('--add_index', action='store_true', default=False, help="Whether to add an index")
parser.add_argument('--collection_name', default="qasper_qwen2_7B_Chunks_310_2", help="Name of the collection") 
parser.add_argument('--retrieve_top_k', type=int, default=5, help="Top k documents to retrieve")

# Milvus Lite 专用参数
parser.add_argument('--milvus_data_dir', default='./milvus_data', help="Milvus Lite data directory (offline storage)")

args = parser.parse_args()
logger.info(args)

# 初始化 LLM
llm = Qwen_7B_Chat(model_name='qwen_7b', temperature=0.1, max_new_tokens=1280)    
# llm = Baichuan2_7B_Chat(model_name='baichuan2_7b', temperature=0.1, max_new_tokens=1280)
# llm = GLM4_9B_Chat(model_name='glm4_9b', temperature=0.1, max_new_tokens=1280)

# 初始化嵌入模型
embed_model = HuggingfaceEmbeddings(model_name=args.embedding_name)
print('[Milvus Lite] 嵌入模型加载完成...')

# 创建 Milvus Lite 检索器（关键改动）
retriever = BaseRetrieverLite(
    docs_directory=args.docs_path,
    embed_model=embed_model,
    embed_dim=args.embedding_dim,
    construct_index=args.construct_index,
    add_index=args.add_index,
    collection_name=args.collection_name,
    similarity_top_k=args.retrieve_top_k,
    milvus_data_dir=args.milvus_data_dir  # 本地存储目录
)

# 显示存储信息
storage_info = retriever.get_storage_info()
logger.info(f"Milvus Lite 存储信息: {storage_info}")

print('[Milvus Lite] 索引准备完成...')

# 开始检索和生成
retrieval_save_list = []
with open(args.data_path, 'r', encoding='utf-8') as file:  
    for line in file: 
        data = json.loads(line) 
        try:
            # 检索相关文档
            retrieval_prompt = retriever.search_docs(data['input'])
            
            # LLM 生成答案
            llm_ans = llm.request(retrieval_prompt)
            print(data['input'], '\n生成答案:', llm_ans, flush=True)
            
            # 保存结果
            save = {}
            save['_id'] = data['_id']
            save['input'] = data['input']   
            save['llm_ans'] = llm_ans
            save['answers'] = data['answers']
            save['retrieval_list'] = retrieval_prompt
            retrieval_save_list.append(save)
            
        except Exception as e:
            logger.error(f"处理失败: {e}")
            pass

# 保存结果
with open(args.save_file, 'w', encoding='utf-8') as json_file:
    json.dump(retrieval_save_list, json_file, indent=4, ensure_ascii=False)

logger.info(f"✅ 评估完成！结果已保存到 {args.save_file}")

# 使用示例
# CUDA_VISIBLE_DEVICES=4,5 nohup python retrieval_lite.py --construct_index >> qa_results/qasper_qwen2_7B.log 2>&1 &
