import json
import time
import os
import multiprocessing
from typing import Any
from tqdm import tqdm
import tiktoken

INPUT_FILE = "f:/thesis/Meta-Chunking/meta-chunking-dataset/meta-chunking/Original_Dataset/LongBench-main/data/2wikimqa.jsonl"
OUTPUT_DIR = "f:/thesis/Meta-Chunking/Chunk_Result"
CHUNK_TOKEN_SIZE = 1200
CHUNK_OVERLAP_TOKEN_SIZE = 100
SPLIT_BY_CHARACTER = "\n\n"
SPLIT_BY_CHARACTER_ONLY = False
NUM_WORKERS = 4


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def init_tokenizer():
    return tiktoken.get_encoding("cl100k_base")


def chunking_by_token_size(
    tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    print(f"Warning: Chunk exceeds token limit: {len(_tokens)} > {chunk_token_size}")
                    for start in range(0, len(_tokens), chunk_token_size - chunk_overlap_token_size):
                        chunk_content = tokenizer.decode(_tokens[start : start + chunk_token_size])
                        new_chunks.append((min(chunk_token_size, len(_tokens) - start), chunk_content))
                else:
                    new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    for start in range(0, len(_tokens), chunk_token_size - chunk_overlap_token_size):
                        chunk_content = tokenizer.decode(_tokens[start : start + chunk_token_size])
                        new_chunks.append((min(chunk_token_size, len(_tokens) - start), chunk_content))
                else:
                    new_chunks.append((len(_tokens), chunk))
        
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append({
                "tokens": _len,
                "content": chunk.strip(),
                "chunk_order_index": index,
            })
    else:
        for index, start in enumerate(range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)):
            chunk_content = tokenizer.decode(tokens[start : start + chunk_token_size])
            results.append({
                "tokens": min(chunk_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            })
    
    return results


def process_context(context_text, doc_id, tokenizer):
    start_time = time.time()
    
    chunks = chunking_by_token_size(
        tokenizer=tokenizer,
        content=context_text,
        split_by_character=SPLIT_BY_CHARACTER,
        split_by_character_only=SPLIT_BY_CHARACTER_ONLY,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN_SIZE,
        chunk_token_size=CHUNK_TOKEN_SIZE
    )
    
    splits = [[chunk['content'], doc_id] for chunk in chunks]
    
    end_time = time.time()
    
    return {
        'splits': splits,
        'time_cost': end_time - start_time
    }


def process_line(line_data):
    try:
        data = json.loads(line_data)
        doc_id = data.get('_id', '')
        context = data.get('context', '')
        
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


def main():
    create_directory(OUTPUT_DIR)
    
    print(f"Reading input file: {INPUT_FILE}")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
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
        all_splits.extend(result['splits'])
        total_time += result['time_cost']
    
    output_data = {
        "filepath": INPUT_FILE,
        "splits": all_splits,
        "time_cost": total_time
    }
    
    input_basename = os.path.basename(INPUT_FILE).replace('.jsonl', '')
    output_file = os.path.join(OUTPUT_DIR, f"{input_basename}_lightrag_chunk.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(results)} documents")
    print(f"Total splits: {len(all_splits)}")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
