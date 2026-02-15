import os
import json
import numpy as np
import logging
from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据集对应的评估指标映射
dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
    # 默认使用 QA F1
    "default": qa_f1_score 
}

class Config:
    # 评估配置
    PREDICTION_FILE = 'F:/thesis/Meta-Chunking/eval/LongBench/sample_results.json'
    OUTPUT_FILE = 'F:/thesis/Meta-Chunking/eval/LongBench/eval_results.json'
    DATASET_NAME = 'qasper'  # 根据实际数据集选择，或者使用 'default'
    EVAL_LONGBENCH_E = False # 是否使用 LongBench-E 评估模式 (分长度统计)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    """
    LongBench-E 评估模式：按长度分段统计分数
    """
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    metric_func = dataset2metric.get(dataset, dataset2metric['default'])
    
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
            
        for ground_truth in ground_truths:
            score = max(score, metric_func(prediction, ground_truth, all_classes=all_classes))
            
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
            
    for key in scores.keys():
        if scores[key]:
            scores[key] = round(100 * np.mean(scores[key]), 2)
        else:
            scores[key] = 0.0
            
    return scores

def scorer(dataset, predictions, answers, all_classes):
    """
    标准评估模式：计算平均分
    """
    total_score = 0.
    metric_func = dataset2metric.get(dataset, dataset2metric['default'])
    
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
            
        for ground_truth in ground_truths:
            score = max(score, metric_func(prediction, ground_truth, all_classes=all_classes))
        total_score += score
        
    return round(100 * total_score / len(predictions), 2)

def main():
    logger.info("开始运行评估脚本...")
    logger.info(f"读取预测文件: {Config.PREDICTION_FILE}")
    
    if not os.path.exists(Config.PREDICTION_FILE):
        logger.error(f"文件不存在: {Config.PREDICTION_FILE}")
        return

    try:
        with open(Config.PREDICTION_FILE, 'r', encoding='utf-8') as file:  
            qa_data = json.load(file)
    except json.JSONDecodeError:
        logger.error(f"JSON 解析失败: {Config.PREDICTION_FILE}")
        return

    predictions, answers, lengths = [], [], []
    
    for qa in qa_data:
        # 兼容不同的字段名
        pred = qa.get('llm_ans', qa.get('prediction', ''))
        ans = qa.get('answers', qa.get('ground_truth', []))
        length = qa.get('length', 0)
        
        predictions.append(pred)
        answers.append(ans)
        lengths.append(length)

    all_classes = None # 如果是分类任务，需要从数据中提取所有类别
    
    logger.info(f"数据集: {Config.DATASET_NAME}, 样本数量: {len(predictions)}")

    scores = dict()
    if Config.EVAL_LONGBENCH_E:
        logger.info("使用 LongBench-E 模式评估 (分长度统计)...")
        score = scorer_e(Config.DATASET_NAME, predictions, answers, lengths, all_classes)
    else:
        logger.info("使用标准模式评估...")
        score = scorer(Config.DATASET_NAME, predictions, answers, all_classes)
    
    scores[Config.DATASET_NAME] = score
    
    # 保存结果
    os.makedirs(os.path.dirname(Config.OUTPUT_FILE), exist_ok=True)
    with open(Config.OUTPUT_FILE, "w", encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
        
    logger.info(f"✅ 评估完成！分数: {score}")
    logger.info(f"结果已保存到: {Config.OUTPUT_FILE}")

if __name__ == '__main__':
    main()
