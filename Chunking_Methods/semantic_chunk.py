import json
import time
import os
import sys
import multiprocessing
from pathlib import Path
from functools import partial
from threading import RLock
from tqdm import tqdm
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document

SERVER_DIR = Path(__file__).resolve().parents[1] / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from app.core.model_factory import get_llamaindex_embedding

# 配置
INPUT_FILE = "/data/h50056789/Rag_Chunking/Corpus/LongBench/2wikimqa.jsonl"
OUTPUT_DIR = "/data/h50056789/Rag_Chunking/Chunk_Result/Semantic_Chunk"
EMBED_MODEL_PATH = "/data/h50056789/Rag_chunk_bench/model/bge-large-en-v1.5"
BUFFER_SIZE = 1
BREAKPOINT_THRESHOLD = 74
NUM_WORKERS = 1

_SPLITTER_CACHE = {}
_CACHE_LOCK = RLock()

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def _get_embed_model(embed_model_path=EMBED_MODEL_PATH):
    return get_llamaindex_embedding(model_path=str(embed_model_path).strip())


def init_splitter():
    return init_splitter_with_params()


def init_splitter_with_params(embed_model_path=EMBED_MODEL_PATH, buffer_size=BUFFER_SIZE, breakpoint_threshold=BREAKPOINT_THRESHOLD):
    cache_key = (str(embed_model_path).strip(), int(buffer_size), int(breakpoint_threshold))
    with _CACHE_LOCK:
        cached = _SPLITTER_CACHE.get(cache_key)
        if cached is not None:
            return cached

        embed_model = _get_embed_model(embed_model_path)
        splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_threshold,
            embed_model=embed_model
        )
        _SPLITTER_CACHE[cache_key] = splitter
        return splitter


def process_context(context_text, doc_id, splitter):
    start_time = time.time()
    
    doc = Document(text=context_text)
    nodes = splitter.get_nodes_from_documents([doc], show_progress=False)
    
    splits = []
    for chunk_id, node in enumerate(nodes, start=1):
        node_text = node.text if hasattr(node, 'text') else node.get_content()
        if doc_id:
            splits.append([node_text, doc_id, chunk_id])
        else:
            splits.append([node_text])
    
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
        splitter = init_splitter()
        result = process_context(context, doc_id, splitter)
        
        return result
    except Exception as e:
        print(f"Error processing line: {e}")
        return None


def _process_line_with_params(line_data, embed_model_path, buffer_size, breakpoint_threshold):
    """
    模块级函数，用于 multiprocessing，避免 pickle 局部函数的问题。
    """
    try:
        data = json.loads(line_data)
        doc_id = data.get('_id', '')
        context = data.get('context', '')
        
        if not context:
            return None
        
        splitter = init_splitter_with_params(
            embed_model_path=embed_model_path,
            buffer_size=buffer_size,
            breakpoint_threshold=breakpoint_threshold
        )
        return process_context(context, doc_id, splitter)
    except Exception as e:
        print(f"Error processing line: {e}")
        return None


def chunk_file(input_file: str, output_dir: str, embed_model_path: str = EMBED_MODEL_PATH, buffer_size: int = BUFFER_SIZE, breakpoint_threshold: int = BREAKPOINT_THRESHOLD, num_workers: int = NUM_WORKERS):
    """
    对文件进行语义分块处理
    
    Args:
        input_file: 输入文件路径（必填）
        output_dir: 输出目录路径（必填）
        embed_model_path: 嵌入模型路径（可选，默认使用全局EMBED_MODEL_PATH）
        buffer_size: 缓冲区大小（可选，默认使用全局BUFFER_SIZE）
        breakpoint_threshold: 断点阈值（可选，默认使用全局BREAKPOINT_THRESHOLD）
        num_workers: 工作进程数（可选，默认使用全局NUM_WORKERS）
    
    Returns:
        dict: {"success": bool, "output_file": str, "message": str}
    """
    try:
        create_directory(output_dir)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        process_func = partial(
            _process_line_with_params,
            embed_model_path=embed_model_path,
            buffer_size=buffer_size,
            breakpoint_threshold=breakpoint_threshold
        )
        
        # 仅传递轻量参数，worker 内部按参数获取缓存后的 splitter
        
        results = []
        if int(num_workers) <= 1:
            splitter = init_splitter_with_params(
                embed_model_path=embed_model_path,
                buffer_size=buffer_size,
                breakpoint_threshold=breakpoint_threshold
            )
            for line_data in tqdm(lines, total=len(lines)):
                try:
                    data = json.loads(line_data)
                    doc_id = data.get('_id', '')
                    context = data.get('context', '')
                    if not context:
                        continue
                    result = process_context(context, doc_id, splitter)
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing line: {e}")
        else:
            with multiprocessing.Pool(processes=num_workers) as pool:
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
        output_file = os.path.join(output_dir, f"{input_basename}_semantic_chunk.json")
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


def chunk_text(text_input: str, embed_model_path: str = EMBED_MODEL_PATH, buffer_size: int = BUFFER_SIZE, breakpoint_threshold: int = BREAKPOINT_THRESHOLD, num_workers: int = NUM_WORKERS):
    """
    对文本进行语义分块处理
    
    Args:
        text_input: 输入文本内容（必填）
        embed_model_path: 嵌入模型路径（可选，默认使用全局EMBED_MODEL_PATH）
        buffer_size: 缓冲区大小（可选，默认使用全局BUFFER_SIZE）
        breakpoint_threshold: 断点阈值（可选，默认使用全局BREAKPOINT_THRESHOLD）
        num_workers: 工作进程数（对单个文本处理不使用，保留参数以保持接口一致性）
    
    Returns:
        dict: {"success": bool, "splits": [[text], ...], "time_cost": float, "message": str}
    """
    try:
        splitter = init_splitter_with_params(embed_model_path=embed_model_path, buffer_size=buffer_size, breakpoint_threshold=breakpoint_threshold)
        
        result = process_context(text_input, None, splitter)
        
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
    output_file = os.path.join(OUTPUT_DIR, f"{input_basename}_semantic_chunk.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(results)} documents")
    print(f"Total splits: {len(all_splits)}")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
