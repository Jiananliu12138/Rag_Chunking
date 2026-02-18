"""
使用 Milvus 的检索脚本（适配 Docker 版本）
有可能跟tiktok有关哦,报错可以研究一下tiktok
"""
import os
import json
import asyncio
import logging
from tqdm import tqdm
from base_lite import BaseRetrieverLite
from embeddings.base import HuggingfaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from llms.base import BaseLLM
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cache_dir = "/data/h50056789/Rag_Chunking/tiktoken_cache" 
os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir

class Config:
    DATA_PATH = '/data/h50056789/Rag_Chunking/QA_corpus/2wikimqa_qa.jsonl'
    SAVE_DIR = '/data/h50056789/Rag_Chunking/Retrival_Result'

    EMBEDDING_NAME = '/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5'
    EMBEDDING_DIM = 1024

    COLLECTION_NAME = "lumber_chunk"
    RETRIEVE_TOP_K = 5
    
    MILVUS_DATA_DIR = '/data/h50056789/Rag_Chunking/milvus_data'

# ============================================================
# LLM 类定义
# ============================================================
class Qwen_7B_Chat(BaseLLM):
    def __init__(self, model_name='qwen_7b', temperature=1.0, max_new_tokens=1024):
        super().__init__(model_name, temperature, max_new_tokens)
        # 直接使用传入的 model_name 作为路径，或者硬编码为正确的本地路径
        # 注意：这里假设 model_name 参数就是本地路径
        local_path = model_name 
        print(f"Loading model from: {local_path}")
        
        # 尝试显式指定 tokenizer 类
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_path, 
                trust_remote_code=True,
                use_fast=False
            )
        except ValueError:
            # 如果 AutoTokenizer 失败，尝试直接使用 Qwen2Tokenizer
            # 注意：这需要 transformers 版本支持 Qwen2
            from transformers import Qwen2Tokenizer
            self.tokenizer = Qwen2Tokenizer.from_pretrained(
                local_path,
                trust_remote_code=True
            )
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
        # 显式生成 attention_mask
        inputs = self.tokenizer(query, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()
        attention_mask = inputs.attention_mask.cuda()
        
        output = self.model.generate(
            input_ids, 
            attention_mask=attention_mask, # 传入 attention_mask
            pad_token_id=self.tokenizer.eos_token_id, # 显式设置 pad_token_id
            **self.gen_kwargs
        )[0]
        
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
            local_path, use_fast=True, trust_remote_code=True)
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

# ============================================================
# 主逻辑
# ============================================================
async def main():
    logger.info("开始运行检索评估脚本...")
    logger.info(f"配置信息: DATA_PATH={Config.DATA_PATH}, COLLECTION={Config.COLLECTION_NAME}")
    logger.info("注意: 索引构建请使用 embed_lite.py")

    # 1. 初始化 LLM
    # 根据需要取消注释对应的模型
    llm = Qwen_7B_Chat(model_name='/data/h50056789/Rag_Chunking/model/Qwen/Qwen2.5-7B-Instruct', temperature=0.1, max_new_tokens=1280)    
    # llm = Baichuan2_7B_Chat(model_name='baichuan2_7b', temperature=0.1, max_new_tokens=1280)
    # llm = GLM4_9B_Chat(model_name='glm4_9b', temperature=0.1, max_new_tokens=1280)
    
    embed_model = HuggingfaceEmbeddings(model_name=Config.EMBEDDING_NAME)
    print('[Milvus] 嵌入模型加载完成...')

    retriever = BaseRetrieverLite(
        embed_model=embed_model,
        embed_dim=Config.EMBEDDING_DIM,
        construct_index=False,
        add_index=False,
        collection_name=Config.COLLECTION_NAME,
        similarity_top_k=Config.RETRIEVE_TOP_K,
        milvus_data_dir=Config.MILVUS_DATA_DIR
    )

    storage_info = retriever.get_storage_info()
    logger.info(f"Milvus 存储信息: {storage_info}")
    
    print('[Milvus] 开始检索...')

    retrieval_save_list = []
    
    try:
        with open(Config.DATA_PATH, 'r', encoding='utf-8') as file:  
            lines = file.readlines()
        
        with tqdm(total=len(lines), desc="检索+生成", unit="问题") as pbar:
            for line in lines: 
                data = json.loads(line) 
                try:
                    pbar.set_postfix({
                        "ID": data['_id'],
                        "Query": data['input'][:10] + "..." if len(data['input']) > 30 else data['input']
                    })
                    
                    response_vector = retriever.search_docs(data['input'])
                    
                    source_nodes = response_vector.source_nodes
                    context_texts = [node.node.get_content() for node in source_nodes]
                    context_str = "\n\n".join(context_texts)
                    
                    prompt = (
                        "Context information is below.\n"
                        "---------------------\n"
                        f"{context_str}\n"
                        "---------------------\n"
                        "Given the context information and not prior knowledge, answer the query.\n"
                        f"Query: {data['input']}\n"
                        "Answer:"
                    )
                    
                    llm_ans = llm.request(prompt)
                    save = {}
                    save['_id'] = data['_id']
                    save['input'] = data['input']   
                    save['llm_ans'] = llm_ans
                    save['answers'] = data['answers']
                    save['retrieval_list'] = context_texts
                    retrieval_save_list.append(save)
                    
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"\n处理单条数据失败 (ID: {data.get('_id', 'unknown')}): {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update(1)

    except FileNotFoundError:
        logger.error(f"找不到数据文件: {Config.DATA_PATH}")
        return

    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    data_basename = os.path.basename(Config.DATA_PATH).replace('.jsonl', '')
    save_filename = f"{data_basename}_{Config.COLLECTION_NAME}.json"
    save_filepath = os.path.join(Config.SAVE_DIR, save_filename)
    
    logger.info(f"\n保存结果文件: {save_filename}")
    
    with open(save_filepath, 'w', encoding='utf-8') as json_file:
        json.dump(retrieval_save_list, json_file, indent=4, ensure_ascii=False)

    logger.info(f"\n✅ 评估完成！结果已保存到 {save_filepath}")

if __name__ == "__main__":
    # 解决事件循环问题
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        print("提示: 建议安装 nest_asyncio 以避免事件循环错误 (pip install nest_asyncio)")

    asyncio.run(main())
