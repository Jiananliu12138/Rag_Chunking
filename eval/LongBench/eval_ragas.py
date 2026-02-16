import os
import json
import logging
import torch
import numpy as np
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
    ContextEntityRecall
)
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    # 评估配置
    PREDICTION_FILE = '/data/h50056789/Rag_Chunking/eval/LongBench/sample_results.json'
    OUTPUT_FILE = '/data/h50056789/Rag_Chunking/eval/LongBench/eval_results.json'
    
    # 模型路径
    LLM_PATH = '/data/h50056789/Rag_Chunking/model/Qwen/Qwen2.5-7B-Instruct'
    EMBEDDING_PATH = '/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5'
    
    # RAGAS 配置
    RAGAS_METRICS = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextRecall(),
        ContextPrecision(),
        ContextEntityRecall()
    ]

class RagasEvaluator:
    def __init__(self, config):
        self.config = config
        self.llm = None
        self.embeddings = None
        self._init_models()

    def _init_models(self):
        """初始化本地 LLM 和 Embedding 模型用于 RAGAS"""
        logger.info(f"Loading LLM from {self.config.LLM_PATH}...")
        try:
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_PATH, trust_remote_code=True)
            
            # 配置显存分配：LLM 和 Embedding 放在 GPU 1 (Card 2)
            # 动态计算显存限制 (例如 90%)
            try:
                gpu_id = 1
                total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3) # GiB
                utilization = 0.90 # 设置显存利用率 90%
                max_mem_gib = int(total_mem * utilization)
                max_mem_str = f"{max_mem_gib}GiB"
                logger.info(f"Setting max memory for GPU {gpu_id} to {max_mem_str} (Total: {total_mem:.2f}GiB, Util: {utilization*100}%)")
            except Exception as e:
                logger.warning(f"Failed to get GPU properties: {e}, using default 46GiB")
                max_mem_str = "46GiB"

            max_memory_mapping = {0: "0GiB", 1: max_mem_str} 
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.LLM_PATH, 
                device_map="auto", 
                max_memory=max_memory_mapping,
                torch_dtype="auto",
                trust_remote_code=True
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                repetition_penalty=1.1
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe, batch_size=4)
            
            logger.info(f"Loading Embeddings from {self.config.EMBEDDING_PATH}...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_PATH,
                model_kwargs={'device': 'cuda:1'}, # 放在 GPU 1
                encode_kwargs={'batch_size': 16} # 增加 embedding 的批处理大小
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise e

    def calculate_ragas_metrics(self, data):
        """计算 RAGAS 指标"""
        logger.info("Calculating RAGAS metrics...")
        
        ragas_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for item in data:
            ragas_data["question"].append(item["input"])
            ragas_data["answer"].append(item["llm_ans"])
            ragas_data["contexts"].append(item.get("retrieval_list", []))
            ragas_data["ground_truth"].append(" ".join(item["answers"])) 
            
        dataset = Dataset.from_dict(ragas_data)
        
        try:
            results = evaluate(
                dataset=dataset,
                metrics=self.config.RAGAS_METRICS,
                llm=self.llm,
                embeddings=self.embeddings,
            )
            return results
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def run(self):
        # 1. 读取数据
        if not os.path.exists(self.config.PREDICTION_FILE):
            logger.error(f"File not found: {self.config.PREDICTION_FILE}")
            return

        with open(self.config.PREDICTION_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 2. 计算 RAGAS 指标
        ragas_results = self.calculate_ragas_metrics(data)
        
        # 3. 读取现有结果并更新（如果存在）
        final_results = {}
        if os.path.exists(self.config.OUTPUT_FILE):
            try:
                with open(self.config.OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    final_results = json.load(f)
            except:
                pass
        
        final_results.update(ragas_results)
        
        # 4. 输出结果
        logger.info("Final RAGAS Evaluation Results:")
        print(json.dumps(final_results, indent=4))
        
        os.makedirs(os.path.dirname(self.config.OUTPUT_FILE), exist_ok=True)
        with open(self.config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
            def convert(o):
                if isinstance(o, np.float32) or isinstance(o, np.float64):
                    return float(o)
                raise TypeError
            json.dump(final_results, f, indent=4, default=convert)
            
        logger.info(f"Results updated in {self.config.OUTPUT_FILE}")

if __name__ == '__main__':
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
        
    evaluator = RagasEvaluator(Config)
    evaluator.run()
