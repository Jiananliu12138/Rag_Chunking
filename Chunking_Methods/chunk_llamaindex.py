import json
import time
import os
import multiprocessing
from functools import partial
from tqdm import tqdm
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Document

INPUT_FILE = "/data/h50056789/Rag_Chunking/Corpus/LongBench/2wikimqa.jsonl" 
OUTPUT_DIR = "/data/h50056789/Rag_Chunking/Chunk_Result/Llamaindex_Chunk"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
NUM_WORKERS = 4


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def init_parser(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, cache_dir: str = None):
    """
    初始化解析器
    
    Args:
        chunk_size: 分块大小（可选，默认使用全局CHUNK_SIZE）
        chunk_overlap: 分块重叠大小（可选，默认使用全局CHUNK_OVERLAP）
        cache_dir: tiktoken缓存目录（可选，默认不设置）
    
    Returns:
        SimpleNodeParser: 配置好的解析器实例
    """
    if cache_dir is not None:
        os.environ["TIKTOKEN_CACHE_DIR"] = "/data/h50056789/Rag_Chunking/tiktoken_cache"
    
    return SimpleNodeParser.from_defaults(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def process_context(context_text, doc_id, parser):
    start_time = time.time()
    
    doc = Document(text=context_text)
    nodes = parser.get_nodes_from_documents([doc], show_progress=False)
    
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
        parser = init_parser()
        result = process_context(context, doc_id, parser)
        
        return result
    except Exception as e:
        print(f"Error processing line: {e}")
        return None


def _process_line_with_parser(line_data, parser):
    """
    模块级函数，用于 multiprocessing，避免 pickle 局部函数的问题。
    """
    try:
        data = json.loads(line_data)
        doc_id = data.get('_id', '')
        context = data.get('context', '')
        
        if not context:
            return None
        return process_context(context, doc_id, parser)
    except Exception as e:
        print(f"Error processing line: {e}")
        return None


def chunk_file(input_file: str, output_dir: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, num_workers: int = NUM_WORKERS, cache_dir: str = None):
    """
    对文件进行分块处理
    
    Args:
        input_file: 输入文件路径（必填）
        output_dir: 输出目录路径（必填）
        chunk_size: 分块大小（可选，默认使用全局CHUNK_SIZE）
        chunk_overlap: 分块重叠大小（可选，默认使用全局CHUNK_OVERLAP）
        num_workers: 工作进程数（可选，默认使用全局NUM_WORKERS）
        cache_dir: tiktoken缓存目录（可选，默认不设置）
    
    Returns:
        dict: {"success": bool, "output_file": str, "message": str}
    """
    try:
        create_directory(output_dir)

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        parser = init_parser(chunk_size=chunk_size, chunk_overlap=chunk_overlap, cache_dir=cache_dir)
        
        # 使用 functools.partial 绑定 parser，避免 pickle 局部函数的问题
        process_func = partial(_process_line_with_parser, parser=parser)
        
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
        output_file = os.path.join(output_dir, f"{input_basename}_llamaindex_chunk.json")
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

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


def chunk_text(text_input: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, num_workers: int = NUM_WORKERS, cache_dir: str = None):
    """
    对文本进行分块处理
    
    Args:
        text_input: 输入文本内容（必填）
        chunk_size: 分块大小（可选，默认使用全局CHUNK_SIZE）
        chunk_overlap: 分块重叠大小（可选，默认使用全局CHUNK_OVERLAP）
        num_workers: 工作进程数（对单个文本处理不使用，保留参数以保持接口一致性）
        cache_dir: tiktoken缓存目录（可选，默认不设置）
    
    Returns:
        dict: {"success": bool, "splits": [[text], ...], "time_cost": float, "message": str}
    """
    try:
        parser = init_parser(chunk_size=chunk_size, chunk_overlap=chunk_overlap, cache_dir=cache_dir)
        
        result = process_context(text_input, None, parser)
        
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
    output_file = os.path.join(OUTPUT_DIR, f"{input_basename}_llamaindex_chunk.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(results)} documents")
    print(f"Total splits: {len(all_splits)}")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
