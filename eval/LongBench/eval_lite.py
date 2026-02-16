import os
import json
import numpy as np
import logging
import asyncio
from typing import List, Dict, Any
from datasets import Dataset 
from ragas import evaluate, RunConfig
from ragas.metrics.collections import (
    context_precision,
    context_recall,
    context_entity_recall,
    answer_relevancy,
    faithfulness,
)
# from langchain_community.llms import HuggingFacePipeline # 已弃用
# from langchain_community.embeddings import HuggingFaceEmbeddings # 已弃用
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from bert_score import score as bert_score
from rouge import Rouge
from metrics_lite import qa_f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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
    ENABLE_RAGAS = True
    RAGAS_METRICS = [
        context_precision,
        context_recall,
        context_entity_recall,
        answer_relevancy,
        faithfulness
    ]

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.llm = None
        self.embeddings = None
        
        if self.config.ENABLE_RAGAS:
            self._init_models()

    def _init_models(self):
        """初始化本地 LLM 和 Embedding 模型用于 RAGAS"""
        logger.info(f"Loading LLM from {self.config.LLM_PATH}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_PATH, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.LLM_PATH, 
                device_map="auto", 
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
            
            # RAGAS v0.1+ 直接接受 LangChain 的 BaseLLM 和 BaseEmbeddings
            # 不需要再用 LangchainLLM 包装
            # 显式传入 batch_size 以启用批量推理，消除 GPU 顺序执行警告
            self.llm = HuggingFacePipeline(pipeline=pipe, batch_size=8)
            
            logger.info(f"Loading Embeddings from {self.config.EMBEDDING_PATH}...")
            # 同样为 Embeddings 设置 batch_size (如果支持的话，通常通过 model_kwargs 或 encode_kwargs)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_PATH,
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'batch_size': 32} # 增加 embedding 的批处理大小
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.config.ENABLE_RAGAS = False

    def calculate_traditional_metrics(self, predictions, answers):
        """计算传统指标 (F1, ROUGE, BERTScore)"""
        logger.info("Calculating traditional metrics...")
        scores = {}
        
        # 1. LongBench Metrics (F1 / ROUGE based on dataset)
        f1_scores = []
        rouge_l_scores = []
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_3_scores = []
        bleu_4_scores = []
        
        rouge = Rouge()
        smooth = SmoothingFunction().method1 # 用于 BLEU 平滑
        
        for pred, ground_truths in zip(predictions, answers):
            # F1 按词是不是一样
            f1 = 0
            for gt in ground_truths:
                f1 = max(f1, qa_f1_score(pred, gt))
            f1_scores.append(f1)
            
            # ROUGE-L 最长序列
            r_l = 0
            for gt in ground_truths:
                try:
                    if not pred.strip() or not gt.strip():
                        continue
                    r_score = rouge.get_scores(pred, gt)[0]['rouge-l']['f']
                    r_l = max(r_l, r_score)
                except:
                    pass
            rouge_l_scores.append(r_l)

            # BLEU 1-4
            try:
                pred_tokens = pred.split()
                refs_tokens = [gt.split() for gt in ground_truths]
                
                # BLEU-1 (weights=(1, 0, 0, 0))
                b1 = sentence_bleu(refs_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
                bleu_1_scores.append(b1)
                
                # BLEU-2 (weights=(0.5, 0.5, 0, 0))
                b2 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
                bleu_2_scores.append(b2)
                
                # BLEU-3 (weights=(0.33, 0.33, 0.33, 0))
                b3 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smooth)
                bleu_3_scores.append(b3)
                
                # BLEU-4 (weights=(0.25, 0.25, 0.25, 0.25))
                b4 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
                bleu_4_scores.append(b4)
                
            except:
                bleu_1_scores.append(0.0)
                bleu_2_scores.append(0.0)
                bleu_3_scores.append(0.0)
                bleu_4_scores.append(0.0)
            
        scores['f1'] = np.mean(f1_scores)
        scores['rouge_l'] = np.mean(rouge_l_scores)
        scores['bleu_1'] = np.mean(bleu_1_scores)
        scores['bleu_2'] = np.mean(bleu_2_scores)
        scores['bleu_3'] = np.mean(bleu_3_scores)
        scores['bleu_4'] = np.mean(bleu_4_scores)
        
        # 2. BERTScore
        logger.info("Calculating BERTScore...")
        try:
            # BERTScore 不支持直接的多参考答案取 max，需要手动处理
            # 策略：对每个样本，计算预测值与所有参考答案的 BERTScore，取 F1 最大的那个
            
            all_f1_scores = []
            
            refs = [" ".join(gts) for gts in answers]
            
            # 加载 BERTScore 的本地模型
            # 注意：bert_score 需要 transformer 模型路径，而不是 sentence-transformer 路径
            # 如果 BGE 模型是基于 BERT 架构的，可以直接用
            # 为了确保加载成功，我们使用 transformers 加载 tokenizer 和 model
            
            # 使用 lang="en" 让它使用默认模型，或者传入本地路径
            # 如果传入路径失败，可能是因为 bert_score 内部对路径的处理问题
            # 尝试直接传入模型路径，并确保路径正确
            model_path = self.config.EMBEDDING_PATH
            
            P, R, F1 = bert_score(
                predictions, 
                refs, 
                model_type=model_path,
                num_layers=None,
                verbose=False, 
                device='cuda',
                batch_size=32 # 显式设置 batch size
            )
            scores['bert_score_f1'] = F1.mean().item()
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            scores['bert_score_f1'] = 0.0
            
        return scores

    def calculate_ragas_metrics(self, data):
        """计算 RAGAS 指标"""
        if not self.config.ENABLE_RAGAS:
            return {}
            
        logger.info("Calculating RAGAS metrics...")
        
        # 准备 RAGAS 数据集格式
        # RAGAS 需要: question, answer, contexts, ground_truth
        ragas_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for item in data:
            ragas_data["question"].append(item["input"])
            ragas_data["answer"].append(item["llm_ans"])
            # contexts 必须是 list[str]
            ragas_data["contexts"].append(item.get("retrieval_list", []))
            # ground_truth 必须是 str (RAGAS v0.1+)
            # 将所有标准答案拼接，让 RAGAS 自己去判断
            ragas_data["ground_truth"].append(" ".join(item["answers"])) 
            
        dataset = Dataset.from_dict(ragas_data)
        
        try:
            results = evaluate(
                dataset=dataset,
                metrics=self.config.RAGAS_METRICS,
                llm=self.llm,
                embeddings=self.embeddings,
                # 降低并发数，增加超时时间
                run_config=RunConfig(max_workers=1, timeout=360) 
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
            
        predictions = [item['llm_ans'] for item in data]
        answers = [item['answers'] for item in data]
        
        # 2. 计算传统指标
        final_results = self.calculate_traditional_metrics(predictions, answers)
        
        # 3. 计算 RAGAS 指标
        ragas_results = self.calculate_ragas_metrics(data)
        final_results.update(ragas_results)
        
        # 4. 输出结果
        logger.info("Final Evaluation Results:")
        print(json.dumps(final_results, indent=4))
        
        os.makedirs(os.path.dirname(self.config.OUTPUT_FILE), exist_ok=True)
        with open(self.config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # 将 numpy 类型转换为 float
            def convert(o):
                if isinstance(o, np.float32) or isinstance(o, np.float64):
                    return float(o)
                raise TypeError
            json.dump(final_results, f, indent=4, default=convert)
            
        logger.info(f"Results saved to {self.config.OUTPUT_FILE}")

if __name__ == '__main__':
    # 解决事件循环问题
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
        
    evaluator = Evaluator(Config)
    evaluator.run()
