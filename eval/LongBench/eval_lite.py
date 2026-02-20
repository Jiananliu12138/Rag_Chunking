import gc
import torch
import os
import json
import numpy as np
import logging
from pathlib import Path
from bert_score import score as bert_score
from rouge import Rouge
from metrics_lite import qa_f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    PREDICTION_FILE = '/data/h50056789/Rag_Chunking/eval/LongBench/sample_results.json'
    OUTPUT_DIR = '/data/h50056789/Rag_Chunking/eval/LongBench'

class Evaluator:
    def __init__(self, config):
        self.config = config

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
            os.environ['HF_HOME'] = '/data/h50056789/Rag_Chunking/model/model_cache'

            all_f1_scores = []
            
            refs = [" ".join(gts) for gts in answers]
            
            P, R, F1 = bert_score(
                predictions, 
                refs, 
                model_type="roberta-large",
                num_layers=None,
                verbose=True, 
                device='cuda:0', # 放在 GPU 0 (Card 1)
                batch_size=16 # 显式设置 batch size
            )
            scores['bert_score_f1'] = F1.mean().item()
            
            # 释放 BERTScore 显存
            del P, R, F1
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("BERTScore memory cleared.")
            
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            scores['bert_score_f1'] = 0.0
            
        return scores

    def run(self, input_json_path=None, output_json_path=None):
        """
        执行评估流程。
        
        Args:
            input_json_path: 输入 JSON 文件路径（可选，默认使用 config.PREDICTION_FILE）
            output_json_path: 输出 JSON 文件路径（可选，默认在输入文件同目录下生成）
        """
        # 1. 确定输入路径
        if input_json_path is None:
            input_json_path = self.config.PREDICTION_FILE
        
        input_path = Path(input_json_path)
        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            return

        # 2. 读取数据
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        predictions = [item['llm_ans'] for item in data]
        answers = [item['answers'] for item in data]
        
        # 3. 计算传统指标
        final_results = self.calculate_traditional_metrics(predictions, answers)
        
        # 4. 确定输出路径
        if output_json_path is None:
            output_json_path = input_path.parent / f"{input_path.stem}_traditional_eval.json"
        else:
            output_json_path = Path(output_json_path)
        
        # 5. 创建输出目录
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 6. 合并已有结果（如果存在）
        if output_json_path.exists():
            try:
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                    existing_results.update(final_results)
                    final_results = existing_results
            except Exception:
                pass
        
        # 7. 保存结果
        print(f"\n{'='*70}")
        print(f"💾 保存评估结果")
        print(f"{'='*70}")
        print(f"输出文件: {output_json_path}")
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            def convert(o):
                if isinstance(o, np.float32) or isinstance(o, np.float64):
                    return float(o)
                raise TypeError
            json.dump(final_results, f, indent=4, default=convert, ensure_ascii=False)
        
        print(f"✅ 评估结果已保存")
        print(f"{'='*70}\n")
        logger.info(f"Results saved to {output_json_path}")

if __name__ == '__main__':
    evaluator = Evaluator(Config)
    evaluator.run()
