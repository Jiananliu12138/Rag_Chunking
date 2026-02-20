import json
import time
import os
import multiprocessing
from functools import partial
from typing import Any
from tqdm import tqdm
import tiktoken

INPUT_FILE = "/data/h50056789/Rag_Chunking/Corpus/LongBench/2wikimqa.jsonl" 
OUTPUT_DIR = "/data/h50056789/Rag_Chunking/Chunk_Result/Lightrag_Chunk"
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
    cache_dir = "/data/h50056789/Rag_Chunking/tiktoken_cache" 
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir 
    return tiktoken.get_encoding("o200k_base")


def init_tokenizer_with_params(cache_dir: str = None):
    if cache_dir is not None:
        os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    return tiktoken.get_encoding("o200k_base")


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


def process_context_with_params(context_text, doc_id, tokenizer, split_by_character=None, split_by_character_only=False, chunk_overlap_token_size=100, chunk_token_size=1200):
    start_time = time.time()
    
    chunks = chunking_by_token_size(
        tokenizer=tokenizer,
        content=context_text,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        chunk_overlap_token_size=chunk_overlap_token_size,
        chunk_token_size=chunk_token_size
    )
    
    if doc_id:
        splits = [[chunk['content'], doc_id] for chunk in chunks]
    else:
        splits = [[chunk['content']] for chunk in chunks]
    
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


def _process_line_with_params(line_data, tokenizer, split_by_character, split_by_character_only, chunk_overlap_token_size, chunk_token_size):
    """
    模块级函数，用于 multiprocessing，避免 pickle 局部函数的问题。
    """
    try:
        data = json.loads(line_data)
        doc_id = data.get('_id', '')
        context = data.get('context', '')
        
        if not context:
            return None
        
        return process_context_with_params(context, doc_id, tokenizer, split_by_character, split_by_character_only, chunk_overlap_token_size, chunk_token_size)
    except Exception as e:
        print(f"Error processing line: {e}")
        return None


def chunk_file(input_file: str, output_dir: str, chunk_token_size: int = CHUNK_TOKEN_SIZE, chunk_overlap_token_size: int = CHUNK_OVERLAP_TOKEN_SIZE, split_by_character: str = SPLIT_BY_CHARACTER, split_by_character_only: bool = SPLIT_BY_CHARACTER_ONLY, num_workers: int = NUM_WORKERS, cache_dir: str = None):
    """
    对文件进行基于Token的分块处理
    
    Args:
        input_file: 输入文件路径（必填）
        output_dir: 输出目录路径（必填）
        chunk_token_size: 分块Token大小（可选，默认使用全局CHUNK_TOKEN_SIZE）
        chunk_overlap_token_size: 分块重叠Token大小（可选，默认使用全局CHUNK_OVERLAP_TOKEN_SIZE）
        split_by_character: 分割字符（可选，默认使用全局SPLIT_BY_CHARACTER）
        split_by_character_only: 是否仅按字符分割（可选，默认使用全局SPLIT_BY_CHARACTER_ONLY）
        num_workers: 工作进程数（可选，默认使用全局NUM_WORKERS）
        cache_dir: tiktoken缓存目录（可选，默认不设置）
    
    Returns:
        dict: {"success": bool, "output_file": str, "message": str}
    """
    try:
        create_directory(output_dir)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        tokenizer = init_tokenizer_with_params(cache_dir=cache_dir)
        
        # 使用 functools.partial 绑定参数，避免 pickle 局部函数的问题
        process_func = partial(
            _process_line_with_params,
            tokenizer=tokenizer,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            chunk_overlap_token_size=chunk_overlap_token_size,
            chunk_token_size=chunk_token_size
        )
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(process_func, lines), total=len(lines)):
                if result:
                    results.append(result)
        
        all_splits = []
        total_time = 0
        for result in results:
            all_splits.extend(result['splits'])
            total_time += result['time_cost']
        
        output_data = {
            "filepath": input_file,
            "splits": all_splits,
            "time_cost": total_time
        }
        
        input_basename = os.path.basename(input_file).replace('.jsonl', '')
        output_file = os.path.join(output_dir, f"{input_basename}_lightrag_chunk.json")
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nProcessed {len(results)} documents")
        print(f"Total splits: {len(all_splits)}")
        print(f"Results saved to: {output_file}")
        
        return {
            "success": True,
            "output_file": output_file,
            "message": f"Successfully processed {len(results)} documents with {len(all_splits)} splits"
        }
    
    except Exception as e:
        return {
            "success": False,
            "output_file": "",
            "message": f"Error: {str(e)}"
        }


def chunk_text(text_input: str, chunk_token_size: int = CHUNK_TOKEN_SIZE, chunk_overlap_token_size: int = CHUNK_OVERLAP_TOKEN_SIZE, split_by_character: str = SPLIT_BY_CHARACTER, split_by_character_only: bool = SPLIT_BY_CHARACTER_ONLY, num_workers: int = NUM_WORKERS, cache_dir: str = None):
    """
    对文本进行基于Token的分块处理
    
    Args:
        text_input: 输入文本内容（必填）
        chunk_token_size: 分块Token大小（可选，默认使用全局CHUNK_TOKEN_SIZE）
        chunk_overlap_token_size: 分块重叠Token大小（可选，默认使用全局CHUNK_OVERLAP_TOKEN_SIZE）
        split_by_character: 分割字符（可选，默认使用全局SPLIT_BY_CHARACTER）
        split_by_character_only: 是否仅按字符分割（可选，默认使用全局SPLIT_BY_CHARACTER_ONLY）
        num_workers: 工作进程数（对单个文本处理不使用，保留参数以保持接口一致性）
        cache_dir: tiktoken缓存目录（可选，默认不设置）
    
    Returns:
        dict: {"success": bool, "splits": [[text], ...], "time_cost": float, "message": str}
    """
    try:
        tokenizer = init_tokenizer_with_params(cache_dir=cache_dir)
        
        result = process_context_with_params(text_input, None, tokenizer, split_by_character, split_by_character_only, chunk_overlap_token_size, chunk_token_size)
        
        return {
            "success": True,
            "splits": result['splits'],
            "time_cost": result['time_cost'],
            "message": f"Successfully chunked text into {len(result['splits'])} splits"
        }
    
    except Exception as e:
        print(f"Error processing text: {e}")
        return {
            "success": False,
            "splits": [],
            "time_cost": 0,
            "message": f"Error: {str(e)}"
        }


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
