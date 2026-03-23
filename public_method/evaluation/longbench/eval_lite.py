import gc
import json
import logging
import os
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from bert_score import BERTScorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge

from .metrics_lite import qa_f1_score


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_BERT_SCORER_CACHE: dict[tuple[str, str, int | None, int], BERTScorer] = {}
_BERT_SCORER_CACHE_LOCK = Lock()


class Config:
    PREDICTION_FILE = "/data/h50056789/Rag_Chunking/Retrival_Result/2wikimqa_qa_semantic_chunk.json"
    OUTPUT_DIR = "/data/h50056789/Rag_Chunking/eval_result"


def _get_bert_scorer(
    model_type: str,
    device: str,
    num_layers: int | None = None,
    batch_size: int = 16,
) -> BERTScorer:
    cache_key = (
        str(model_type).strip(),
        str(device).strip(),
        num_layers,
        int(batch_size),
    )
    with _BERT_SCORER_CACHE_LOCK:
        cached = _BERT_SCORER_CACHE.get(cache_key)
        if cached is not None:
            return cached

        scorer = BERTScorer(
            model_type=model_type,
            num_layers=num_layers,
            device=device,
            batch_size=batch_size,
        )
        _BERT_SCORER_CACHE[cache_key] = scorer
        return scorer


def bert_score(
    predictions: list[str],
    refs: list[str],
    model_type: str,
    num_layers: int | None = None,
    verbose: bool = True,
    device: str = "cuda:0",
    batch_size: int = 16,
):
    scorer = _get_bert_scorer(
        model_type=model_type,
        device=device,
        num_layers=num_layers,
        batch_size=batch_size,
    )
    return scorer.score(
        predictions,
        refs,
        verbose=verbose,
        batch_size=batch_size,
    )


def _calculate_metric_scores(predictions: list[str], answers: list[list[str]]) -> dict:
    scores = {}
    f1_scores = []
    rouge_l_scores = []
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []

    rouge = Rouge()
    smooth = SmoothingFunction().method1

    for pred, ground_truths in zip(predictions, answers):
        f1 = 0.0
        for gt in ground_truths:
            f1 = max(f1, qa_f1_score(pred, gt))
        f1_scores.append(f1)

        rouge_l = 0.0
        for gt in ground_truths:
            try:
                if not pred.strip() or not gt.strip():
                    continue
                rouge_score = rouge.get_scores(pred, gt)[0]["rouge-l"]["f"]
                rouge_l = max(rouge_l, rouge_score)
            except Exception:
                pass
        rouge_l_scores.append(rouge_l)

        try:
            pred_tokens = pred.split()
            refs_tokens = [gt.split() for gt in ground_truths]

            bleu_1_scores.append(
                sentence_bleu(refs_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
            )
            bleu_2_scores.append(
                sentence_bleu(refs_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
            )
            bleu_3_scores.append(
                sentence_bleu(refs_tokens, pred_tokens, weights=(0.333, 0.333, 0.333, 0), smoothing_function=smooth)
            )
            bleu_4_scores.append(
                sentence_bleu(refs_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
            )
        except Exception:
            bleu_1_scores.append(0.0)
            bleu_2_scores.append(0.0)
            bleu_3_scores.append(0.0)
            bleu_4_scores.append(0.0)

    scores["f1"] = np.mean(f1_scores)
    scores["rouge_l"] = np.mean(rouge_l_scores)
    scores["bleu_1"] = np.mean(bleu_1_scores)
    scores["bleu_2"] = np.mean(bleu_2_scores)
    scores["bleu_3"] = np.mean(bleu_3_scores)
    scores["bleu_4"] = np.mean(bleu_4_scores)
    return scores


def calculate_traditional_metrics_with_params(
    predictions: list[str],
    answers: list[list[str]],
    enable_bert_score: bool = True,
    bert_score_model: str = "roberta-large",
    bert_score_device: str = "cuda:0",
    hf_home: str | None = None,
) -> dict:
    logger.info("Calculating traditional metrics...")
    scores = _calculate_metric_scores(predictions, answers)

    if enable_bert_score:
        logger.info("Calculating BERTScore...")
        try:
            if hf_home:
                os.environ["HF_HOME"] = hf_home

            refs = [" ".join(gts) for gts in answers]
            _, _, f1 = bert_score(
                predictions,
                refs,
                model_type=bert_score_model,
                num_layers=None,
                verbose=True,
                device=bert_score_device,
                batch_size=16,
            )
            scores["bert_score_f1"] = f1.mean().item()

            del f1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("BERTScore memory cleared.")
        except Exception as exc:
            logger.warning("BERTScore calculation failed: %s", exc)
            scores["bert_score_f1"] = 0.0
    else:
        scores["bert_score_f1"] = None

    return scores


class Evaluator:
    def __init__(self, config):
        self.config = config

    def calculate_traditional_metrics(self, predictions, answers):
        return calculate_traditional_metrics_with_params(
            predictions=predictions,
            answers=answers,
            enable_bert_score=True,
            bert_score_model="roberta-large",
            bert_score_device="cuda:0",
            hf_home="/data/h50056789/Rag_Chunking/model/model_cache",
        )

    def run(self, input_json_path=None, output_json_path=None):
        if input_json_path is None:
            input_json_path = self.config.PREDICTION_FILE

        input_path = Path(input_json_path)
        if not input_path.exists():
            logger.error("File not found: %s", input_path)
            return

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        predictions = [item["llm_ans"] for item in data]
        answers = [item["answers"] for item in data]

        final_results = self.calculate_traditional_metrics(predictions, answers)

        if output_json_path is None:
            output_json_path = input_path.parent / f"{input_path.stem}_traditional_eval.json"
        else:
            output_json_path = Path(output_json_path)

        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        if output_json_path.exists():
            try:
                with open(output_json_path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                existing_results.update(final_results)
                final_results = existing_results
            except Exception:
                pass

        print(f"\n{'=' * 70}")
        print("Saving evaluation results")
        print(f"{'=' * 70}")
        print(f"Output file: {output_json_path}")

        with open(output_json_path, "w", encoding="utf-8") as f:
            def convert(obj):
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                raise TypeError

            json.dump(final_results, f, indent=4, default=convert, ensure_ascii=False)

        logger.info("Results saved to %s", output_json_path)


if __name__ == "__main__":
    evaluator = Evaluator(Config)
    evaluator.run()
