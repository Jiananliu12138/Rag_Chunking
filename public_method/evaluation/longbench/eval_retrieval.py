import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    INPUT_PATH: str = r"F:\thesis\Meta-Chunking\public_method\evaluation\longbench\sample_results_api.json"
    OUTPUT_PATH: str = r"F:\thesis\Meta-Chunking\public_method\evaluation\longbench\sample_results_api_retrieval_eval.json"
    CUTS: tuple[int, ...] = (1, 3, 5, 10)
    SKIP_EMPTY_GOLD: bool = True


class RetrievalEvaluator:
    """
    Evaluate retrieval quality from RAG output file.
    Expected row format:
    {
      "_id": ...,
      "rag_retrieval": [{"text","filepath","doc_id","chunk_id","score?"}, ...],
      "gold_reference": [{"text","filepath","doc_id","chunk_id"}, ...]
    }
    """

    def __init__(self, cuts: tuple[int, ...], skip_empty_gold: bool = True):
        self._cuts = tuple(sorted(set(int(k) for k in cuts if int(k) > 0)))
        self._skip_empty_gold = skip_empty_gold

    @staticmethod
    def _load_rows(input_path: str) -> list[dict[str, Any]]:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"input file not found: {input_path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            return [data]
        raise ValueError("input must be json object or json list")

    @staticmethod
    def _stable_text_key(text: str) -> str:
        normalized = " ".join(text.strip().split())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

    @classmethod
    def _doc_key(
        cls,
        *,
        doc_id: Optional[Any],
        chunk_id: Optional[Any],
        filepath: Optional[Any],
        text: Optional[Any],
    ) -> Optional[str]:
        did = str(doc_id).strip() if doc_id is not None and str(doc_id).strip() else None
        cid = str(chunk_id).strip() if chunk_id is not None and str(chunk_id).strip() else None
        fpath = str(filepath).strip() if filepath is not None and str(filepath).strip() else None
        txt = str(text).strip() if text is not None and str(text).strip() else None

        if did and cid:
            return f"doc:{did}#chunk:{cid}"
        if did:
            return f"doc:{did}"
        if fpath and cid:
            return f"path:{fpath}#chunk:{cid}"
        if fpath:
            return f"path:{fpath}"
        if txt:
            return f"text:{cls._stable_text_key(txt)}"
        return None

    @staticmethod
    def _qid(row: dict[str, Any], row_idx: int) -> str:
        raw = row.get("_id")
        if raw is None:
            return str(row_idx)
        if isinstance(raw, str) and not raw.strip():
            return str(row_idx)
        return str(raw)

    @staticmethod
    def _dcg(binary_rels: list[int], k: int) -> float:
        val = 0.0
        for i, rel in enumerate(binary_rels[:k], start=1):
            if rel <= 0:
                continue
            val += 1.0 / math.log2(i + 1.0)
        return val

    def _metrics_for_query(self, gold_set: set[str], ranked_doc_ids: list[str]) -> dict[str, float]:
        num_rel = len(gold_set)
        binary = [1 if d in gold_set else 0 for d in ranked_doc_ids]
        out: dict[str, float] = {}

        # AP / MAP
        if num_rel == 0:
            out["map"] = 0.0
        else:
            hit = 0
            ap_sum = 0.0
            for i, rel in enumerate(binary, start=1):
                if rel:
                    hit += 1
                    ap_sum += hit / i
            out["map"] = ap_sum / num_rel

        # MRR
        rr = 0.0
        for i, rel in enumerate(binary, start=1):
            if rel:
                rr = 1.0 / i
                break
        out["recip_rank"] = rr

        # R-precision
        if num_rel == 0:
            out["Rprec"] = 0.0
        else:
            out["Rprec"] = sum(binary[:num_rel]) / num_rel

        # nDCG
        if num_rel == 0:
            out["ndcg"] = 0.0
        else:
            dcg = self._dcg(binary, len(binary))
            idcg = self._dcg([1] * num_rel, num_rel)
            out["ndcg"] = (dcg / idcg) if idcg > 0 else 0.0

        # P@k / Recall@k / nDCG@k
        for k in self._cuts:
            topk = binary[:k]
            out[f"P_{k}"] = sum(topk) / k
            out[f"recall_{k}"] = (sum(topk) / num_rel) if num_rel > 0 else 0.0
            if num_rel == 0:
                out[f"ndcg_cut_{k}"] = 0.0
            else:
                dcg_k = self._dcg(binary, k)
                idcg_k = self._dcg([1] * min(num_rel, k), k)
                out[f"ndcg_cut_{k}"] = (dcg_k / idcg_k) if idcg_k > 0 else 0.0
        return out

    @staticmethod
    def _aggregate(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
        by_metric: dict[str, list[float]] = {}
        for q_metrics in per_query.values():
            for metric, value in q_metrics.items():
                by_metric.setdefault(metric, []).append(float(value))
        return {metric: mean(vals) for metric, vals in by_metric.items()}

    def evaluate_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        per_query: dict[str, dict[str, float]] = {}
        diagnostics: list[dict[str, Any]] = []
        skipped_empty_gold = 0

        for idx, row in enumerate(tqdm(rows, desc="Evaluate retrieval", unit="query"), start=1):
            qid = self._qid(row, idx)
            raw_gold = row.get("gold_reference") if isinstance(row.get("gold_reference"), list) else []
            raw_ret = row.get("rag_retrieval") if isinstance(row.get("rag_retrieval"), list) else []

            gold_keys: set[str] = set()
            for item in raw_gold:
                if not isinstance(item, dict):
                    continue
                key = self._doc_key(
                    doc_id=item.get("doc_id"),
                    chunk_id=item.get("chunk_id"),
                    filepath=item.get("filepath"),
                    text=item.get("text"),
                )
                if key:
                    gold_keys.add(key)

            if self._skip_empty_gold and not gold_keys:
                skipped_empty_gold += 1
                continue

            ret_pairs: list[tuple[str, float]] = []
            default_score = float(len(raw_ret))
            for rank, item in enumerate(raw_ret):
                if not isinstance(item, dict):
                    continue
                key = self._doc_key(
                    doc_id=item.get("doc_id"),
                    chunk_id=item.get("chunk_id"),
                    filepath=item.get("filepath"),
                    text=item.get("text"),
                )
                if not key:
                    continue
                score = item.get("score")
                score_val = float(score) if isinstance(score, (int, float)) else (default_score - rank)
                ret_pairs.append((key, score_val))

            # stable ranking: score desc, key asc
            ret_pairs.sort(key=lambda x: (-x[1], x[0]))
            ranked_doc_ids = [x[0] for x in ret_pairs]

            metrics = self._metrics_for_query(gold_set=gold_keys, ranked_doc_ids=ranked_doc_ids)
            per_query[qid] = metrics

            overlap = len(set(ranked_doc_ids) & gold_keys)
            diagnostics.append(
                {
                    "query_id": qid,
                    "num_gold": len(gold_keys),
                    "num_retrieved": len(ranked_doc_ids),
                    "num_overlap": overlap,
                }
            )

        if not per_query:
            raise ValueError("no valid query to evaluate")

        return {
            "meta": {
                "total_rows": len(rows),
                "evaluated_queries": len(per_query),
                "skipped_empty_gold": skipped_empty_gold,
                "cuts": list(self._cuts),
            },
            "aggregated": self._aggregate(per_query),
            "per_query": per_query,
            "diagnostics": diagnostics,
        }

    def evaluate_file(self, input_path: str, output_path: str) -> dict[str, Any]:
        rows = self._load_rows(input_path)
        result = self.evaluate_rows(rows)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


def main() -> None:
    evaluator = RetrievalEvaluator(cuts=Config.CUTS, skip_empty_gold=Config.SKIP_EMPTY_GOLD)
    logger.info("Input: %s", Config.INPUT_PATH)
    logger.info("Output: %s", Config.OUTPUT_PATH)
    result = evaluator.evaluate_file(Config.INPUT_PATH, Config.OUTPUT_PATH)
    logger.info("Evaluated queries: %s", result["meta"]["evaluated_queries"])
    for k in ("map", "recip_rank", "ndcg", "Rprec"):
        if k in result["aggregated"]:
            logger.info("%s=%.6f", k, result["aggregated"][k])


if __name__ == "__main__":
    main()
