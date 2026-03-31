import json
import multiprocessing
import os
import re
import time
from functools import partial
from typing import Any

import tiktoken
from tqdm import tqdm

INPUT_FILE = "/data/h50056789/Rag_Chunking/Corpus/LongBench/2wikimqa.jsonl"
OUTPUT_DIR = "/data/h50056789/Rag_Chunking/Chunk_Result/Lightrag_Chunk"
CHUNK_TOKEN_SIZE = 1200
CHUNK_OVERLAP_TOKEN_SIZE = 100
MIN_CHUNK_TOKENS = 0

# Frontend input examples:
# - 2wikimqa / hotpotqa:
#   Split by Character = (?=Passage\s+\d+:)
#   Use Regex = True
# - narrativeqa:
#   Split by Character = \n\n\n\n
#   Use Regex = False
# - If NarrativeQA still produces dirty chunks from title pages / transcriber
#   notes / other front matter, trim that prefix before chunking.
SPLIT_BY_CHARACTER = "\n\n"
SPLIT_USE_REGEX = False
SPLIT_BY_CHARACTER_ONLY = False
NUM_WORKERS = 4


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def init_tokenizer():
    cache_dir = "/data/h50056789/Rag_Chunking/tiktoken_cache"
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    return tiktoken.get_encoding("o200k_base")


def init_tokenizer_with_params(cache_dir: str = None):
    if cache_dir is not None:
        os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    return tiktoken.get_encoding("o200k_base")


def _split_content(
    content: str,
    split_by_character: str | None,
    split_use_regex: bool,
) -> list[str]:
    if not split_by_character:
        return [content]

    if split_use_regex:
        raw_chunks = re.split(split_by_character, content)
    else:
        raw_chunks = content.split(split_by_character)

    chunks = [chunk for chunk in raw_chunks if isinstance(chunk, str) and chunk.strip()]
    return chunks or [content]


def _get_merge_separator(
    split_by_character: str | None,
    split_use_regex: bool,
) -> str:
    if split_by_character and not split_use_regex:
        return split_by_character
    return "\n\n"


def _merge_short_chunks(
    chunks: list[dict[str, Any]],
    min_chunk_tokens: int,
    merge_separator: str,
) -> list[dict[str, Any]]:
    if min_chunk_tokens <= 0 or len(chunks) <= 1:
        return chunks

    merged: list[dict[str, Any]] = []
    pending_texts: list[str] = []
    pending_tokens = 0

    def flush_pending() -> None:
        nonlocal pending_texts, pending_tokens
        if not pending_texts:
            return
        merged.append(
            {
                "tokens": pending_tokens,
                "content": merge_separator.join(pending_texts).strip(),
                "chunk_order_index": len(merged),
            }
        )
        pending_texts = []
        pending_tokens = 0

    for chunk in chunks:
        content = str(chunk.get("content") or "").strip()
        token_count = int(chunk.get("tokens") or 0)
        if not content:
            continue

        if pending_texts:
            pending_texts.append(content)
            pending_tokens += token_count
            if pending_tokens >= min_chunk_tokens:
                flush_pending()
            continue

        if token_count < min_chunk_tokens:
            pending_texts = [content]
            pending_tokens = token_count
            continue

        merged.append(
            {
                "tokens": token_count,
                "content": content,
                "chunk_order_index": len(merged),
            }
        )

    if pending_texts:
        if merged:
            merged[-1]["content"] = (
                f"{merged[-1]['content']}{merge_separator}{merge_separator.join(pending_texts)}"
            ).strip()
            merged[-1]["tokens"] += pending_tokens
        else:
            flush_pending()

    for index, chunk in enumerate(merged):
        chunk["chunk_order_index"] = index
    return merged


def chunking_by_token_size(
    tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_use_regex: bool = False,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
    min_chunk_tokens: int = 0,
) -> list[dict[str, Any]]:
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []

    if split_by_character:
        raw_chunks = _split_content(content, split_by_character, split_use_regex)
        new_chunks = []

        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    print(
                        f"Warning: Chunk exceeds token limit: {len(_tokens)} > {chunk_token_size}"
                    )
                    for start in range(
                        0,
                        len(_tokens),
                        chunk_token_size - chunk_overlap_token_size,
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start : start + chunk_token_size]
                        )
                        new_chunks.append(
                            (min(chunk_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    for start in range(
                        0,
                        len(_tokens),
                        chunk_token_size - chunk_overlap_token_size,
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start : start + chunk_token_size]
                        )
                        new_chunks.append(
                            (min(chunk_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))

        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + chunk_token_size])
            results.append(
                {
                    "tokens": min(chunk_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )

    merge_separator = _get_merge_separator(split_by_character, split_use_regex)
    results = _merge_short_chunks(results, min_chunk_tokens, merge_separator)
    return results


def process_context(context_text, doc_id, tokenizer):
    start_time = time.time()

    chunks = chunking_by_token_size(
        tokenizer=tokenizer,
        content=context_text,
        split_by_character=SPLIT_BY_CHARACTER,
        split_use_regex=SPLIT_USE_REGEX,
        split_by_character_only=SPLIT_BY_CHARACTER_ONLY,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN_SIZE,
        chunk_token_size=CHUNK_TOKEN_SIZE,
        min_chunk_tokens=MIN_CHUNK_TOKENS,
    )

    if doc_id:
        splits = [
            [chunk["content"], doc_id, chunk_id]
            for chunk_id, chunk in enumerate(chunks, start=1)
        ]
    else:
        splits = [[chunk["content"]] for chunk in chunks]

    end_time = time.time()

    return {
        "splits": splits,
        "time_cost": end_time - start_time,
    }


def process_context_with_params(
    context_text,
    doc_id,
    tokenizer,
    split_by_character=None,
    split_use_regex=False,
    split_by_character_only=False,
    chunk_overlap_token_size=100,
    chunk_token_size=1200,
    min_chunk_tokens=0,
):
    start_time = time.time()

    chunks = chunking_by_token_size(
        tokenizer=tokenizer,
        content=context_text,
        split_by_character=split_by_character,
        split_use_regex=split_use_regex,
        split_by_character_only=split_by_character_only,
        chunk_overlap_token_size=chunk_overlap_token_size,
        chunk_token_size=chunk_token_size,
        min_chunk_tokens=min_chunk_tokens,
    )

    if doc_id:
        splits = [
            [chunk["content"], doc_id, chunk_id]
            for chunk_id, chunk in enumerate(chunks, start=1)
        ]
    else:
        splits = [[chunk["content"]] for chunk in chunks]

    end_time = time.time()

    return {
        "splits": splits,
        "time_cost": end_time - start_time,
    }


def process_line(line_data):
    try:
        data = json.loads(line_data)
        doc_id = data.get("_id", "")
        context = data.get("context", "")

        if not context:
            print(f"Skip empty context for ID: {doc_id}")
            return None

        print(f"Processing ID: {doc_id}")
        tokenizer = init_tokenizer()
        result = process_context(context, doc_id, tokenizer)

        return result
    except Exception as e:
        print(f"Error processing line: {e}")
        return None


def _process_line_with_params(
    line_data,
    tokenizer,
    split_by_character,
    split_use_regex,
    split_by_character_only,
    chunk_overlap_token_size,
    chunk_token_size,
    min_chunk_tokens,
):
    try:
        data = json.loads(line_data)
        doc_id = data.get("_id", "")
        context = data.get("context", "")

        if not context:
            return None

        return process_context_with_params(
            context,
            doc_id,
            tokenizer,
            split_by_character,
            split_use_regex,
            split_by_character_only,
            chunk_overlap_token_size,
            chunk_token_size,
            min_chunk_tokens,
        )
    except Exception as e:
        print(f"Error processing line: {e}")
        return None


def chunk_file(
    input_file: str,
    output_dir: str,
    chunk_token_size: int = CHUNK_TOKEN_SIZE,
    chunk_overlap_token_size: int = CHUNK_OVERLAP_TOKEN_SIZE,
    split_by_character: str = SPLIT_BY_CHARACTER,
    split_use_regex: bool = SPLIT_USE_REGEX,
    split_by_character_only: bool = SPLIT_BY_CHARACTER_ONLY,
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
    num_workers: int = NUM_WORKERS,
    cache_dir: str = None,
):
    """
    Split a JSONL file into chunks with optional literal or regex pre-splitting.
    """
    try:
        create_directory(output_dir)

        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        tokenizer = init_tokenizer_with_params(cache_dir=cache_dir)

        process_func = partial(
            _process_line_with_params,
            tokenizer=tokenizer,
            split_by_character=split_by_character,
            split_use_regex=split_use_regex,
            split_by_character_only=split_by_character_only,
            chunk_overlap_token_size=chunk_overlap_token_size,
            chunk_token_size=chunk_token_size,
            min_chunk_tokens=min_chunk_tokens,
        )

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(process_func, lines), total=len(lines)):
                if result:
                    results.append(result)

        all_splits = []
        total_time = 0
        for result in results:
            all_splits.extend(result["splits"])
            total_time += result["time_cost"]

        output_data = {
            "filepath": input_file,
            "splits": all_splits,
            "time_cost": total_time,
        }

        input_basename = os.path.basename(input_file).replace(".jsonl", "")
        output_file = os.path.join(output_dir, f"{input_basename}_lightrag_chunk.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nProcessed {len(results)} documents")
        print(f"Total splits: {len(all_splits)}")
        print(f"Results saved to: {output_file}")

        return {
            "success": True,
            "output_file": output_file,
            "message": f"Successfully processed {len(results)} documents with {len(all_splits)} splits",
        }

    except Exception as e:
        return {
            "success": False,
            "output_file": "",
            "message": f"Error: {str(e)}",
        }


def chunk_text(
    text_input: str,
    chunk_token_size: int = CHUNK_TOKEN_SIZE,
    chunk_overlap_token_size: int = CHUNK_OVERLAP_TOKEN_SIZE,
    split_by_character: str = SPLIT_BY_CHARACTER,
    split_use_regex: bool = SPLIT_USE_REGEX,
    split_by_character_only: bool = SPLIT_BY_CHARACTER_ONLY,
    min_chunk_tokens: int = MIN_CHUNK_TOKENS,
    num_workers: int = NUM_WORKERS,
    cache_dir: str = None,
):
    """
    Split a text string with optional literal or regex pre-splitting.
    """
    try:
        tokenizer = init_tokenizer_with_params(cache_dir=cache_dir)

        result = process_context_with_params(
            text_input,
            None,
            tokenizer,
            split_by_character,
            split_use_regex,
            split_by_character_only,
            chunk_overlap_token_size,
            chunk_token_size,
            min_chunk_tokens,
        )

        return {
            "success": True,
            "splits": result["splits"],
            "time_cost": result["time_cost"],
            "message": f"Successfully chunked text into {len(result['splits'])} splits",
        }

    except Exception as e:
        print(f"Error processing text: {e}")
        return {
            "success": False,
            "splits": [],
            "time_cost": 0,
            "message": f"Error: {str(e)}",
        }


def main():
    create_directory(OUTPUT_DIR)

    print(f"Reading input file: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Total lines: {len(lines)}")

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(process_line, lines), total=len(lines)):
            if result:
                results.append(result)

    all_splits = []
    total_time = 0
    for result in results:
        all_splits.extend(result["splits"])
        total_time += result["time_cost"]

    output_data = {
        "filepath": INPUT_FILE,
        "splits": all_splits,
        "time_cost": total_time,
    }

    input_basename = os.path.basename(INPUT_FILE).replace(".jsonl", "")
    output_file = os.path.join(OUTPUT_DIR, f"{input_basename}_lightrag_chunk.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nProcessed {len(results)} documents")
    print(f"Total splits: {len(all_splits)}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
