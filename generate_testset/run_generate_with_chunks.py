import json
import logging
import statistics
import os
import math

from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
from urllib import request

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from openai import AsyncOpenAI

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.testset.transforms import (
    CosineSimilarityBuilder,
    EmbeddingExtractor,
    OverlapScoreBuilder,
    Parallel,
    SummaryExtractor,
)
from ragas.testset.transforms.extractors.llm_based import (
    NERExtractor,
    ThemesExtractor,
)
os.environ["TIKTOKEN_CACHE_DIR"] = "/data/h50056789/Rag_Chunking/tiktoken_cache"
Chunk = Union[str, Document]

# ===== Static config (edit directly) =====
CHUNKS_FILE = Path("/data/h50056789/Rag_Chunking/test_database/2wikimqa_llamaindex_chunk.json")
OUTPUT_FILE = Path("/data/h50056789/Rag_Chunking/test_database/rag_testset.jsonl")
TESTSET_SIZE = 10
WITH_DEBUGGING_LOGS = True

LLM_BASE_URL = "http://127.0.0.1:8001/v1"
LLM_API_KEY = "EMPTY"
LLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
LLM_PROVIDER = "openai"
LLM_ADAPTER = "auto"
LLM_TIMEOUT_SECONDS = 120
LLM_PREFLIGHT_TIMEOUT_SECONDS = 8

MAX_WORKERS = 4
MAX_RETRIES = 3
MAX_WAIT_SECONDS = 20
RUN_TIMEOUT_SECONDS = 120
LLM_MAX_TOKENS = 4096
FAIL_FAST = False

# In many local-vLLM runs, CustomNodeFilter produces excessive "no summary" logs
# and does not filter effectively for pre-chunked inputs. Keep this True by default.
SKIP_CUSTOM_NODE_FILTER = True
QUIET_FILTER_WARNINGS = True

EMBEDDING_MODEL_PATH = Path(r"/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5")
EMBEDDING_DEVICE = "cuda"

CHUNK_META_PREFIX = "<<<MC_META>>>"
CHUNK_META_SUFFIX = "<<<END_MC_META>>>"


def _load_chunks(path: Path) -> List[Chunk]:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "splits" in data:
            source_filepath = str(data.get("filepath") or "")
            splits = data.get("splits")
            if not isinstance(splits, list):
                raise ValueError("`splits` in JSON must be a list")
            return _normalize_split_tuples(splits, source_filepath)
        if not isinstance(data, list):
            raise ValueError("JSON must be a list, or a dict containing `splits`")
        return _normalize_chunks(data)

    if suffix == ".jsonl":
        rows: List[Any] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
        return _normalize_chunks(rows)

    raise ValueError(f"Unsupported chunks file format: {suffix}")


def _normalize_split_tuples(
    splits: Sequence[Any],
    source_filepath: str = "",
) -> List[Chunk]:
    normalized: List[Chunk] = []
    for item in splits:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            raise TypeError("Each split item must be [chunk_text, article_id, chunk_id]")

        chunk_text = item[0]
        article_id = item[1]
        chunk_id = item[2]
        if not isinstance(chunk_text, str):
            raise TypeError("split[0] must be a string")
        text = chunk_text.strip()
        if not text:
            continue

        metadata = {
            "source_article_id": article_id,
            "source_chunk_id": chunk_id,
        }
        if source_filepath:
            metadata["source_filepath"] = source_filepath

        text_with_meta = _inject_chunk_meta_into_text(
            text=text,
            source_article_id=article_id,
            source_chunk_id=chunk_id,
            source_filepath=source_filepath or None,
        )
        normalized.append(Document(page_content=text_with_meta, metadata=metadata))

    return normalized


def _normalize_chunks(items: Sequence[Any]) -> List[Chunk]:
    normalized: List[Chunk] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append(text)
            continue

        if isinstance(item, dict):
            page_content = item.get("page_content") or item.get("text") or ""
            metadata = item.get("metadata", {})
            if isinstance(page_content, str) and page_content.strip():
                text = page_content.strip()
                source_article_id = metadata.get("source_article_id")
                source_chunk_id = metadata.get("source_chunk_id")
                source_filepath = metadata.get("source_filepath")
                if source_article_id is not None or source_chunk_id is not None or source_filepath:
                    text = _inject_chunk_meta_into_text(
                        text=text,
                        source_article_id=source_article_id,
                        source_chunk_id=source_chunk_id,
                        source_filepath=source_filepath,
                    )
                normalized.append(Document(page_content=text, metadata=metadata))
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 3:
            normalized.extend(_normalize_split_tuples([item]))
            continue

        raise TypeError(
            "Chunk items must be str, dict, or [text, article_id, chunk_id]"
        )

    return normalized


def _inject_chunk_meta_into_text(
    text: str,
    source_article_id: Any = None,
    source_chunk_id: Any = None,
    source_filepath: Any = None,
) -> str:
    payload = {
        "doc_id": source_article_id,
        "chunk_id": source_chunk_id,
        "source_filepath": source_filepath,
    }
    meta_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"{CHUNK_META_PREFIX}{meta_json}{CHUNK_META_SUFFIX}\n{text}"


def extract_all_chunk_meta_from_text(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {"metas": [], "content": text}

    metas: List[Dict[str, Any]] = []
    pieces: List[str] = []
    cursor = 0
    text_len = len(text)

    while cursor < text_len:
        prefix_pos = text.find(CHUNK_META_PREFIX, cursor)
        if prefix_pos == -1:
            pieces.append(text[cursor:])
            break

        pieces.append(text[cursor:prefix_pos])
        suffix_pos = text.find(CHUNK_META_SUFFIX, prefix_pos + len(CHUNK_META_PREFIX))
        if suffix_pos == -1:
            # Broken marker: keep original tail to avoid data loss.
            pieces.append(text[prefix_pos:])
            break

        meta_json = text[prefix_pos + len(CHUNK_META_PREFIX):suffix_pos]
        try:
            meta = json.loads(meta_json)
            if isinstance(meta, dict):
                metas.append(meta)
        except json.JSONDecodeError:
            pass

        cursor = suffix_pos + len(CHUNK_META_SUFFIX)
        if cursor < text_len and text[cursor] == "\n":
            cursor += 1

    content = "".join(pieces).lstrip("\n")
    return {"metas": metas, "content": content}


def _extract_reference_contexts_meta(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        contexts = row.get("reference_contexts")
        if not isinstance(contexts, list):
            continue

        cleaned_contexts: List[Any] = []
        context_metas: List[Any] = []
        for ctx in contexts:
            if not isinstance(ctx, str):
                cleaned_contexts.append(ctx)
                context_metas.append(None)
                continue

            parsed = extract_all_chunk_meta_from_text(ctx)
            cleaned_contexts.append(parsed["content"])
            context_metas.append(parsed["metas"])

        row["reference_contexts"] = cleaned_contexts
        existing_meta = row.get("meta")
        if not isinstance(existing_meta, dict):
            existing_meta = {}
        existing_meta["reference_contexts"] = context_metas
        row["meta"] = existing_meta


def _chunk_text(chunk: Chunk) -> str:
    if isinstance(chunk, Document):
        return chunk.page_content
    return chunk


def _build_llm():
    # Use async client so ragas async transforms do not block the event loop.
    client = AsyncOpenAI(
        api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT_SECONDS
    )
    llm_kwargs = {}
    if LLM_MAX_TOKENS is not None:
        llm_kwargs["max_tokens"] = LLM_MAX_TOKENS
    return llm_factory(
        model=LLM_MODEL,
        provider=LLM_PROVIDER,
        client=client,
        adapter=LLM_ADAPTER,
        **llm_kwargs,
    )


def _build_transforms(llm, embedding_model):
    def filter_chunks(node):
        return node.type.name == "CHUNK"

    summary_extractor = SummaryExtractor(llm=llm, filter_nodes=filter_chunks)
    summary_emb_extractor = EmbeddingExtractor(
        embedding_model=embedding_model,
        property_name="summary_embedding",
        embed_property_name="summary",
        filter_nodes=filter_chunks,
    )
    theme_extractor = ThemesExtractor(llm=llm, filter_nodes=filter_chunks)
    ner_extractor = NERExtractor(llm=llm, filter_nodes=filter_chunks)
    cosine_sim_builder = CosineSimilarityBuilder(
        property_name="summary_embedding",
        new_property_name="summary_similarity",
        threshold=0.7,
        filter_nodes=filter_chunks,
    )
    ner_overlap_sim = OverlapScoreBuilder(threshold=0.01, filter_nodes=filter_chunks)

    transforms = [
        summary_extractor,
        Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
        Parallel(cosine_sim_builder, ner_overlap_sim),
    ]
    return transforms


def _build_embedding_model():
    model_path = EMBEDDING_MODEL_PATH.expanduser()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Local embedding model path not found: {model_path.resolve()}"
        )
    langchain_embeddings = LCHuggingFaceEmbeddings(
        model_name=str(model_path),
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(langchain_embeddings)


def _preflight_check_model_endpoint():
    base = LLM_BASE_URL.rstrip("/")
    models_url = f"{base}/models"
    req = request.Request(
        models_url,
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=LLM_PREFLIGHT_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"LLM endpoint preflight failed: {models_url} is unreachable or invalid. "
            f"Details: {exc}"
        ) from exc

    available = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
    if available and LLM_MODEL not in available:
        print(
            f"[warn] Model '{LLM_MODEL}' not in /models list. "
            f"Available (sample): {sorted(list(available))[:5]}"
        )


def _configure_logging():
    if QUIET_FILTER_WARNINGS:
        logging.getLogger("ragas.testset.transforms.filters").setLevel(logging.ERROR)


def _split_success_and_failed_rows(result: Any) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not hasattr(result, "to_list") or not callable(getattr(result, "to_list")):
        raise TypeError(
            "Expected generation result with callable .to_list(), "
            f"got {type(result).__name__}"
        )

    raw_rows = result.to_list()
    rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []

    for index, row in enumerate(raw_rows):
        if isinstance(row, dict):
            rows.append(row)
            continue

        failed_rows.append(
            {
                "index": index,
                "type": type(row).__name__,
                "repr": repr(row),
            }
        )

    return rows, failed_rows


def _safe_repr(value: Any, max_len: int = 1000) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # pragma: no cover - defensive logging helper
        text = f"<repr failed: {type(exc).__name__}: {exc}>"
    if len(text) > max_len:
        return text[:max_len] + "...<truncated>"
    return text


def _is_nan_like(value: Any) -> bool:
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def _build_failed_generation_record(
    index: int,
    sample: Any,
    additional_info: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "index": index,
        "type": type(sample).__name__,
        "repr": _safe_repr(sample),
        "reason": reason,
    }
    for key, value in additional_info.items():
        if key not in record:
            record[key] = value
    return record


def _patch_ragas_safe_generate() -> None:
    import ragas.testset.synthesizers.generate as ragas_generate

    if getattr(ragas_generate.TestsetGenerator.generate, "_mc_safe_patched", False):
        return

    def _safe_generate(
        self,
        testset_size: int,
        query_distribution: Any = None,
        num_personas: int = 3,
        run_config: Any = None,
        batch_size: Any = None,
        callbacks: Any = None,
        token_usage_parser: Any = None,
        with_debugging_logs: bool = False,
        raise_exceptions: bool = True,
        return_executor: bool = False,
    ) -> Any:
        if run_config is not None and isinstance(self.llm, ragas_generate.BaseRagasLLM):
            self.llm.set_run_config(run_config)

        query_distribution = query_distribution or ragas_generate.default_query_distribution(
            self.llm,
            self.knowledge_graph,
            self.llm_context,
        )
        callbacks = callbacks or []
        ragas_callbacks = {}

        if token_usage_parser is not None:
            from ragas.cost import CostCallbackHandler

            cost_cb = CostCallbackHandler(token_usage_parser=token_usage_parser)
            ragas_callbacks["cost_cb"] = cost_cb
        else:
            cost_cb = None

        for cb in ragas_callbacks.values():
            if isinstance(callbacks, ragas_generate.BaseCallbackManager):
                callbacks.add_handler(cb)
            else:
                callbacks.append(cb)

        testset_generation_rm, testset_generation_grp = ragas_generate.new_group(
            name=ragas_generate.RAGAS_TESTSET_GENERATION_GROUP_NAME,
            inputs={"testset_size": testset_size},
            callbacks=callbacks,
        )

        if with_debugging_logs:
            from ragas.utils import patch_logger

            patch_logger("ragas.experimental.testset.synthesizers", logging.DEBUG)
            patch_logger("ragas.experimental.testset.graph", logging.DEBUG)
            patch_logger("ragas.experimental.testset.transforms", logging.DEBUG)

        if self.persona_list is None:
            self.persona_list = ragas_generate.generate_personas_from_kg(
                llm=self.llm,
                kg=self.knowledge_graph,
                num_personas=num_personas,
                callbacks=callbacks,
            )
        else:
            ragas_generate.random.shuffle(self.persona_list)

        splits, _ = ragas_generate.calculate_split_values(
            [prob for _, prob in query_distribution],
            testset_size,
        )
        scenario_generation_rm, scenario_generation_grp = ragas_generate.new_group(
            name="Scenario Generation",
            inputs={"splits": splits},
            callbacks=testset_generation_grp,
        )

        exec = ragas_generate.Executor(
            desc="Generating Scenarios",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=False,
            batch_size=batch_size,
        )

        splits, _ = ragas_generate.calculate_split_values(
            [prob for _, prob in query_distribution],
            testset_size,
        )
        for i, (scenario, _) in enumerate(query_distribution):
            exec.submit(
                scenario.generate_scenarios,
                n=splits[i],
                knowledge_graph=self.knowledge_graph,
                persona_list=self.persona_list[:num_personas],
                callbacks=scenario_generation_grp,
            )

        try:
            scenario_sample_list = exec.results()
        except Exception as e:
            scenario_generation_rm.on_chain_error(e)
            raise e
        else:
            scenario_generation_rm.on_chain_end(
                outputs={"scenario_sample_list": scenario_sample_list}
            )

        sample_generation_rm, sample_generation_grp = ragas_generate.new_group(
            name="Sample Generation",
            inputs={"scenario_sample_list": scenario_sample_list},
            callbacks=testset_generation_grp,
        )
        exec = ragas_generate.Executor(
            "Generating Samples",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=True,
            batch_size=batch_size,
        )
        additional_testset_info: List[Dict[str, Any]] = []
        for i, (synthesizer, _) in enumerate(query_distribution):
            for sample in scenario_sample_list[i]:
                exec.submit(
                    synthesizer.generate_sample,
                    scenario=sample,
                    callbacks=sample_generation_grp,
                )
                additional_testset_info.append(
                    {
                        "synthesizer_name": synthesizer.name,
                    }
                )

        if return_executor:
            self._mc_last_failed_rows = []
            return exec

        try:
            eval_samples = exec.results()
        except Exception as e:
            sample_generation_rm.on_chain_error(e)
            raise e
        else:
            sample_generation_rm.on_chain_end(outputs={"eval_samples": eval_samples})

        failed_rows: List[Dict[str, Any]] = []
        if len(eval_samples) != len(additional_testset_info):
            failed_rows.append(
                {
                    "index": -1,
                    "type": "length_mismatch",
                    "repr": "",
                    "reason": (
                        "eval_samples and additional_testset_info lengths differ: "
                        f"{len(eval_samples)} vs {len(additional_testset_info)}"
                    ),
                }
            )

        testsets = []
        for index, (sample, additional_info) in enumerate(
            zip(eval_samples, additional_testset_info)
        ):
            if _is_nan_like(sample):
                failed_rows.append(
                    _build_failed_generation_record(
                        index=index,
                        sample=sample,
                        additional_info=additional_info,
                        reason="eval_sample is NaN",
                    )
                )
                continue

            try:
                testsets.append(
                    ragas_generate.TestsetSample(
                        eval_sample=sample,
                        **additional_info,
                    )
                )
            except Exception as exc:
                failed_rows.append(
                    _build_failed_generation_record(
                        index=index,
                        sample=sample,
                        additional_info=additional_info,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )

        self._mc_last_failed_rows = failed_rows

        testset = ragas_generate.Testset(samples=testsets, cost_cb=cost_cb)
        testset_generation_rm.on_chain_end({"testset": testset})
        ragas_generate.track(
            ragas_generate.TestsetGenerationEvent(
                event_type="testset_generation",
                evolution_names=[
                    e.__class__.__name__.lower() for e, _ in query_distribution
                ],
                evolution_percentages=[p for _, p in query_distribution],
                num_rows=len(testsets),
                language="english",
            )
        )
        return testset

    _safe_generate._mc_safe_patched = True
    ragas_generate.TestsetGenerator.generate = _safe_generate


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    _configure_logging()
    _preflight_check_model_endpoint()
    _patch_ragas_safe_generate()

    chunks = _load_chunks(CHUNKS_FILE)

    llm = _build_llm()
    embedding_model = _build_embedding_model()
    transforms = _build_transforms(llm, embedding_model) if SKIP_CUSTOM_NODE_FILTER else None

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embedding_model,
    )
    generator._mc_last_failed_rows = []

    result = generator.generate_with_chunks(
        chunks=chunks,
        testset_size=TESTSET_SIZE,
        transforms=transforms,
        transforms_llm=None,
        transforms_embedding_model=None,
        query_distribution=None,
        run_config=RunConfig(
            timeout=RUN_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
            max_wait=MAX_WAIT_SECONDS,
            max_workers=MAX_WORKERS,
        ),
        callbacks=None,
        token_usage_parser=None,
        with_debugging_logs=WITH_DEBUGGING_LOGS,
        raise_exceptions=FAIL_FAST,
        return_executor=False,
    )
    rows, failed_rows = _split_success_and_failed_rows(result)
    failed_rows.extend(getattr(generator, "_mc_last_failed_rows", []))

    _extract_reference_contexts_meta(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUTPUT_FILE, rows)

    if failed_rows:
        failed_output_file = OUTPUT_FILE.with_name(f"{OUTPUT_FILE.stem}.failed.jsonl")
        _write_jsonl(failed_output_file, failed_rows)
        print(
            f"[warn] Encountered {len(failed_rows)} failed generations. "
            f"Details saved to {failed_output_file.resolve()}"
        )

    if len(rows) < TESTSET_SIZE:
        print(
            f"[warn] Requested {TESTSET_SIZE} samples, generated {len(rows)}. "
            "This usually means some generations failed or scenario candidates were insufficient after graph transforms."
        )
    print(f"Done: {len(rows)} samples -> {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
