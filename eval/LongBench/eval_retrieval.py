import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from tqdm import tqdm

import pytrec_eval


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    # Input supports:
    # 1) retrieval/generate-file output json (list[dict])
    # 2) jsonl where each line is one dict
    INPUT_PATH: str = r"F:\thesis\Meta-Chunking\eval\LongBench\sample_results_api.json"
    OUTPUT_PATH: str = r"F:\thesis\Meta-Chunking\eval\LongBench\sample_results_api_retrieval_eval.json"

    # If True, evaluate all trec metrics defined in pytrec_eval/trec_eval.
    # If False, use MEASURES below.
    USE_ALL_TREC: bool = True

    # Example custom metric set. Ignored when USE_ALL_TREC=True.
    MEASURES: tuple[str, ...] = (
        "map",
        "recip_rank",
        "Rprec",
        "ndcg",
        "ndcg_cut.1,3,5,10",
        "recall.1,3,5,10,20,50",
        "P.1,3,5,10",
    )

    # Skip rows that have no valid gold reference doc key.
    SKIP_EMPTY_GOLD: bool = True


class RetrievalIREvaluator:
    """Evaluate retrieval quality from RAG result files via pytrec_eval."""

    def __init__(
        self,
        measures: set[str],
        use_all_trec: bool = True,
        skip_empty_gold: bool = True,
    ):
        self._measures = {"all_trec"} if use_all_trec else measures
        self._skip_empty_gold = skip_empty_gold

    @staticmethod
    def _load_rows(input_path: str) -> list[dict[str, Any]]:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"input file not found: {input_path}")

        if path.suffix.lower() == ".jsonl":
            rows: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # support a single sample object file
            if "rag_retrieval" in data or "gold_reference" in data:
                return [data]
            # support wrapped payloads: {"data": [...]}
            wrapped = data.get("data")
            if isinstance(wrapped, list):
                return [x for x in wrapped if isinstance(x, dict)]
        raise ValueError("input JSON must be a list or use .jsonl format")

    @staticmethod
    def _normalize_qid(row: dict[str, Any], row_idx: int) -> str:
        # Prefer explicit _id/id when not None and not empty string, otherwise auto index.
        raw_id = row.get("_id", None)
        if raw_id is None:
            raw_id = row.get("id", None)
        if raw_id is None:
            return str(row_idx)
        if isinstance(raw_id, str) and not raw_id.strip():
            return str(row_idx)
        return str(raw_id)

    @staticmethod
    def _stable_text_key(text: str) -> str:
        text_norm = " ".join(text.strip().split())
        return hashlib.sha1(text_norm.encode("utf-8")).hexdigest()

    @classmethod
    def _build_doc_key(
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
    def _ensure_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _extract_gold_items(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        # Preferred normalized field from backend.
        if isinstance(row.get("gold_reference"), list):
            return [x for x in row["gold_reference"] if isinstance(x, dict)]

        # Fallback for raw dataset-like format.
        gold_items: list[dict[str, Any]] = []
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        raw_meta_refs = meta.get("reference_contexts")
        raw_contexts = row.get("reference_contexts")

        meta_ref_flat: list[dict[str, Any]] = []
        if isinstance(raw_meta_refs, list):
            for hop in raw_meta_refs:
                hop_list = hop if isinstance(hop, list) else [hop]
                for item in hop_list:
                    if isinstance(item, dict):
                        meta_ref_flat.append(item)

        contexts = self._ensure_list(raw_contexts)
        for idx, ctx_text in enumerate(contexts):
            hop_meta = meta_ref_flat[idx] if idx < len(meta_ref_flat) else {}
            if not isinstance(hop_meta, dict):
                hop_meta = {}
            gold_items.append(
                {
                    "text": ctx_text if isinstance(ctx_text, str) else "",
                    "filepath": hop_meta.get("source_filepath") or hop_meta.get("filepath"),
                    "doc_id": hop_meta.get("doc_id"),
                    "chunk_id": hop_meta.get("chunk_id"),
                }
            )
        return gold_items

    def _extract_retrieval_items(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        raw = row.get("rag_retrieval")
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
        return []

    def _row_to_qrel_run(
        self,
        qid: str,
        row: dict[str, Any],
    ) -> tuple[dict[str, int], dict[str, float], dict[str, Any]]:
        gold_items = self._extract_gold_items(row)
        ret_items = self._extract_retrieval_items(row)

        qrel: dict[str, int] = {}
        for item in gold_items:
            doc_key = self._build_doc_key(
                doc_id=item.get("doc_id"),
                chunk_id=item.get("chunk_id"),
                filepath=item.get("filepath") or item.get("source_filepath"),
                text=item.get("text"),
            )
            if doc_key:
                qrel[doc_key] = 1

        run: dict[str, float] = {}
        default_rank_score = float(len(ret_items))
        for rank, item in enumerate(ret_items):
            doc_key = self._build_doc_key(
                doc_id=item.get("doc_id"),
                chunk_id=item.get("chunk_id"),
                filepath=item.get("filepath") or item.get("source_filepath"),
                text=item.get("text"),
            )
            if not doc_key:
                continue

            score = item.get("score")
            if isinstance(score, (int, float)):
                run_score = float(score)
            else:
                # keep stable ranking when score is missing.
                run_score = default_rank_score - float(rank)

            if doc_key in run:
                run[doc_key] = max(run[doc_key], run_score)
            else:
                run[doc_key] = run_score

        overlap = set(qrel.keys()) & set(run.keys())
        debug = {
            "query_id": qid,
            "num_gold": len(qrel),
            "num_retrieved": len(run),
            "num_overlap": len(overlap),
        }
        return qrel, run, debug

    @staticmethod
    def _aggregate(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
        metric_values: dict[str, list[float]] = {}
        for q_metrics in per_query.values():
            for metric, value in q_metrics.items():
                metric_values.setdefault(metric, []).append(float(value))
        return {metric: mean(vals) for metric, vals in metric_values.items()}

    def evaluate_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        qrel: dict[str, dict[str, int]] = {}
        run: dict[str, dict[str, float]] = {}
        diagnostics: list[dict[str, Any]] = []

        skipped_empty_gold = 0
        for idx, row in enumerate(tqdm(rows, desc="Build qrel/run", unit="query"), start=1):
            qid = self._normalize_qid(row=row, row_idx=idx)
            qrel_row, run_row, debug = self._row_to_qrel_run(qid=qid, row=row)

            if self._skip_empty_gold and not qrel_row:
                skipped_empty_gold += 1
                continue

            qrel[qid] = qrel_row
            run[qid] = run_row
            diagnostics.append(debug)

        if not qrel:
            raise ValueError("no valid qrel generated; check gold_reference format")

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, self._measures)
        per_query = evaluator.evaluate(run)
        aggregated = self._aggregate(per_query)

        return {
            "meta": {
                "total_rows": len(rows),
                "evaluated_queries": len(per_query),
                "skipped_empty_gold": skipped_empty_gold,
                "requested_measures": sorted(self._measures),
                "resolved_measures": sorted(next(iter(per_query.values())).keys()) if per_query else [],
            },
            "aggregated": aggregated,
            "per_query": per_query,
            "diagnostics": diagnostics,
        }

    def evaluate_file(self, input_path: str, output_path: str) -> dict[str, Any]:
        rows = self._load_rows(input_path)
        result = self.evaluate_rows(rows)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


def main() -> None:
    evaluator = RetrievalIREvaluator(
        measures=set(Config.MEASURES),
        use_all_trec=Config.USE_ALL_TREC,
        skip_empty_gold=Config.SKIP_EMPTY_GOLD,
    )

    logger.info("Input: %s", Config.INPUT_PATH)
    logger.info("Output: %s", Config.OUTPUT_PATH)
    result = evaluator.evaluate_file(Config.INPUT_PATH, Config.OUTPUT_PATH)

    logger.info("Evaluated queries: %s", result["meta"]["evaluated_queries"])
    logger.info("Resolved metrics: %s", len(result["meta"]["resolved_measures"]))
    for k in ("map", "recip_rank", "ndcg", "Rprec"):
        if k in result["aggregated"]:
            logger.info("%s=%.6f", k, result["aggregated"][k])


if __name__ == "__main__":
    main()
